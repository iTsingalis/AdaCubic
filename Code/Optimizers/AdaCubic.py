"""
This module implements adaptive cubic regularization of Newton's method.

Implementation by Ioannis Tsingalis
ioannis.tsingalis@gmail.com

References:
    [1] Nesterov, Y., & Polyak, B. T. (2006). Cubic regularization of Newton method and its global performance.
    Mathematical Programming, 108(1), 177-205.
    [2] Cartis, C., Gould, N. I., & Toint, P. L. (2011). Adaptive cubic regularisation methods for unconstrained optimization.
    Part I: motivation, convergence and numerical results. Mathematical Programming, 127(2), 245-295.
    [3] Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods (Vol. 1). Siam.
    [4] Gould, N. I., Lucidi, S., Roma, M., & Toint, P. L. (1999). Solving the trust-region subproblem using the Lanczos
    method. SIAM Journal on Optimization, 9(2), 504-525.
    [5] https://github.com/iTsingalis/torch-trust-ncg
"""

import torch
from torch import Tensor, norm
from torch.optim.optimizer import Optimizer

import numpy as np

from typing import Tuple

from typing import Union

import warnings


class AdaCubic(Optimizer):

    def __init__(self, params,
                 eta1: float,
                 eta2: float,
                 alpha1: float,
                 alpha2: float,
                 kappa_easy=0.01,
                 grad_tol=1e-4,
                 xi0: float = 0.05,
                 gamma1: float = 0.9,
                 hutchinson_iters=1,
                 average_conv_kernel: bool = False,
                 solver='exact', **kwargs):

        defaults = dict(eta1=eta1, eta2=eta2, alpha1=alpha1, alpha2=alpha2,
                        kappa_easy=kappa_easy, grad_tol=grad_tol, xi0=xi0,
                        hutchinson_iters=hutchinson_iters, gamma1=gamma1,
                        average_conv_kernel=average_conv_kernel, solver=solver, **kwargs)

        assert solver in ['exact'], 'Available optimizers: "exact"'

        super(AdaCubic, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("AdaCubic doesn't support per-parameter options (parameter groups)")

        self.generator = torch.Generator().manual_seed(2147483647)

        for p in self.get_params():
            p.hessian = torch.zeros_like(p.data)

        self.max_xi = torch.tensor(1e+3, device=self.param_groups[0]['params'][0].device)
        self.min_xi = torch.tensor(1e-16, device=self.param_groups[0]['params'][0].device)

    @staticmethod
    def lambda_const(lambda_k):
        """
        A small constant defined in [4], Sec. 5.2.
        """
        return (1 + lambda_k) * torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps))

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return [p for group in self.param_groups for p in group['params'] if p.requires_grad]

    def zero_hessian(self):

        for p in self.get_params():
            if not isinstance(p.hessian, float):
                p.hessian.zero_()

    @torch.no_grad()
    def _quad_model(
            self,
            p: Tensor,
            loss: float,
            gradient: Tensor,
            hess_vp: Tensor,
            nu: Tensor) -> Tensor:
        """
        Returns the value of the local quadratic approximation
        """
        qm = loss + (gradient * p).sum() + 0.5 * (hess_vp * p).sum() + nu * norm(p) ** 3 / 6
        return qm

    @torch.no_grad()
    def _lambda_d_plus(self, H, device):

        lambda_d = torch.min(H)  # .to(device=device)
        u_d = torch.zeros(H.size(dim=0), device=device)  # .to(device=device)
        u_d[torch.argmin(H)] = 1.0

        return torch.max(-lambda_d, torch.zeros_like(lambda_d)), lambda_d, u_d

    @torch.no_grad()
    def calc_boundaries(
            self,
            iterate: Tensor,
            direction: Tensor,
            r: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculates the offset to the boundaries of the trust region
        """

        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - r ** 2
        poly_det = b * b - 4 * a * c
        if poly_det < 0:  # Just in case which does not happen in practice
            poly_det = abs(poly_det)
            warnings.warn("Negative Determinant! eta1: {} eta2: {} beta1: {} beta2 {}"
                          .format(self.defaults['eta1'], self.defaults['eta2'],
                                  self.defaults['beta1'], self.defaults['beta2']))

        sqrt_discriminant = torch.sqrt(poly_det)
        # assert not torch.isnan(sqrt_discriminant), 'nan discriminant'

        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        if ta.item() < tb.item():
            return ta, tb
        else:
            return tb, ta

    def _gather_flat_grad(self) -> Union[Tensor, list]:
        """
        Concatenates all gradients into a single gradient vector
        """
        views = []
        for p in self.get_params():
            if p.grad is not None:
                view = p.grad.view(-1)
                views.append(view)

        output = torch.cat(views, 0)
        return output

    @torch.no_grad()
    def _improvement_ratio(self, p, start_loss, flat_grad, closure, nu):
        """
        Calculates the ratio of the actual to the expected improvement

            Arguments:
                p (torch.tensor): The update vector for the parameters
                start_loss (torch.tensor): The value of the loss function
                    before applying the optimization step
                flat_grad (torch.tensor): The flattened gradient vector of the
                    parameters
                closure (callable): The function that evaluates the loss for
                    the current values of the parameters
            Returns:
                The ratio of the actual improvement of the loss to the expected
                improvement, as predicted by the local quadratic model
        """

        if self.defaults['solver'] == 'exact':
            # Apply the update on the parameter to calculate the loss on the new point
            H = torch.cat([torch.flatten(pr.hessian) for pr in self.get_params()], dim=-1)
            hess_vp = H * p

        # Apply the update of the parameter vectors.
        # Use a torch.no_grad() context since we are updating the parameters in
        # place
        with torch.no_grad():
            start_idx = 0
            for param in self.get_params():
                num_els = param.numel()
                curr_upd = p[start_idx:start_idx + num_els]
                param.data.add_(curr_upd.view_as(param), alpha=1)
                start_idx += num_els
        # No need to backpropagate since we only need the value of the loss at
        # the new point to find the ratio of the actual and the expected
        # improvement
        new_loss = closure(backward=False)
        # The numerator represents the actual loss decrease
        numerator = start_loss - new_loss

        # up_params = torch.cat([torch.flatten(pr.data) for pr in self.get_params()], dim=-1)
        new_quad_val = self._quad_model(p, start_loss, flat_grad, hess_vp, nu)

        # The denominator
        denominator = start_loss - new_quad_val

        # TODO: Convert to epsilon, print warning
        ratio = numerator / (denominator + 1e-20)

        gamma_numerator = (1 - self.defaults['eta2']) * torch.sum(flat_grad * p)
        gamma_denominator = (1 - self.defaults['eta2']) * (start_loss + torch.sum(flat_grad * p)
                                                           + self.defaults['eta2'] * new_quad_val - new_loss)
        gamma_bad = gamma_numerator / (gamma_denominator + 1e-20)
        return ratio, gamma_bad

    @torch.enable_grad()
    def _compute_hessian(self):

        hutchinson_iters = self.defaults['hutchinson_iters']

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            params.append(p)

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(hutchinson_iters):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device, dtype=torch.float32) * 2.0
                  - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True,
                                       retain_graph=i < hutchinson_iters - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hessian += (h_z * z) / hutchinson_iters  # approximate the expected values of z*(H@z)

    @torch.no_grad()
    def _converged(self, s, v):
        if abs(norm(s) - v) <= self.defaults['kappa_easy'] * v:
            return True
        else:
            return False

    @torch.no_grad()
    def _nu_next(self,
                 nu: Tensor,
                 r: Tensor,
                 xi: Tensor,
                 H_x: Tensor,
                 g_x: Tensor) -> Tensor:

        H_x_inv = 1. / (H_x + .5 * nu * r + 1e-20)
        s_x = - H_x_inv * g_x

        h_x_norm = norm(s_x)
        dnu_h_x = - .5 * r * H_x_inv * s_x

        phi = 1. / (h_x_norm + 1e-20) - 1. / xi ** (1. / 3)
        dnu_phi = - (torch.sum(dnu_h_x * s_x)) / (h_x_norm + 1e-20) ** 3

        if dnu_phi < 0:
            print(RuntimeWarning('dr_phi2 < 0 violation'))

        _nu = nu - phi / (dnu_phi + 1e-20)

        return _nu

    @torch.no_grad()
    def _compute_h_x(self, nu, r, lambda_const, lanczos_g, T_x):
        try:
            L = torch.linalg.cholesky(T_x + .5 * nu * r * torch.eye(*T_x.size(), out=torch.empty_like(T_x)))
        except RuntimeError:
            lambda_const *= 2
            # RecursionError: maximum recursion depth exceeded while calling a Python object
            h, L = self._compute_h_x(nu=nu + lambda_const, r=r, lambda_const=lambda_const,
                                     lanczos_g=lanczos_g, T_x=T_x)

        h = torch.cholesky_solve(-lanczos_g, L, upper=False)
        return h, L

    @torch.no_grad()
    def _nu_next_lanczos(self,
                         nu: Tensor,
                         r: Tensor,
                         xi: Tensor,
                         h_x: Tensor,
                         w: Tensor) -> Tuple[Tensor, bool]:

        h_x_norm = norm(h_x)
        norm_w = norm(w)

        phi = 1. / (h_x_norm + 1e-20) - 1. / (xi + 1e-20) ** (1. / 3)
        dnu_phi = .5 * r * norm_w ** 2 / (h_x_norm + 1e-20) ** 3

        if dnu_phi < 1e-8:
            raise ValueError('dnu_phi == 0 violation')

        if dnu_phi < 0:
            raise ValueError('dnu_phi < 0 violation')
            # print(RuntimeWarning('du_phi2 < 0 violation'))

        if phi > 0:
            raise ValueError('phi > 0 violation')
            # print(RuntimeWarning('du_phi2 < 0 violation'))

        _nu = nu - phi / (dnu_phi + 1e-20)
        if _nu < 0:
            raise ValueError('nu < 0 violation')

        return _nu

    @torch.no_grad()
    def _solve_subproblem_exact(
            self,
            loss: float,
            flat_grad: Tensor,
            xi: Tensor) -> Tuple[Tensor, Tensor]:

        H_x = torch.cat([torch.flatten(p.hessian) for p in self.get_params()], dim=-1)
        lambda_k, lambda_d, u_d = self._lambda_d_plus(H_x, device=flat_grad.device)
        lambda_const = AdaCubic.lambda_const(2 * lambda_k)
        r = xi ** (1 / 3)

        if lambda_k == 0:
            # print('Positive definite')
            nu = torch.tensor([0. + lambda_const], device=flat_grad.device, dtype=flat_grad.dtype)
            assert nu * r > 0, 'nu * r must be bigger that 0'
        else:
            nu = 2 * (lambda_k + lambda_const) / r
            assert nu * r > 2 * lambda_k, 'nu * r must be bigger that -2 * lambda_min'

        s_x = - (1. / (H_x + .5 * nu * r + 1e-20)) * flat_grad

        if norm(s_x) ** 3 < xi or (norm(s_x) ** 3 - xi) < 1e-4:
            if lambda_k == 0 or abs(norm(s_x) ** 3 - xi) < 1e-4:
                return nu, s_x

            # print('Inside TR - Interior')
            ta_nu, tb_nu = self.calc_boundaries(iterate=s_x, direction=u_d, r=xi ** (1. / 3))
            pa_nu = s_x + ta_nu * u_d
            pb_nu = s_x + tb_nu * u_d

            # Calculate the point on the boundary with the smallest value
            bound_pa_nu_val = self._quad_model(pa_nu, loss, flat_grad, H_x * pa_nu, nu)
            bound_pb_nu_val = self._quad_model(pb_nu, loss, flat_grad, H_x * pb_nu, nu)

            if bound_pa_nu_val <= bound_pb_nu_val:
                return nu, pa_nu
            else:
                return nu, pb_nu

        iter_nu = 0
        while True:
            if self._converged(s_x, xi ** (1. / 3)) or norm(s_x) < np.finfo(float).eps:
                break

            if iter_nu > 115:  # Increase precision
                print(RuntimeWarning('Max iter reached for nu'))
                break
            nu = self._nu_next(nu, r, xi, H_x, flat_grad)
            s_x = - (1. / (H_x + .5 * nu * r + 1e-20)) * flat_grad
            iter_nu += 1

        return nu, s_x

    def xi(self) -> float:
        return self.state["xi"].detach().cpu().item()

    def improvement_ratio(self) -> float:
        return self.state['improvement_ratio'].detach().cpu().item()

    def step(self, closure=None) -> Tuple[float, str]:
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        starting_loss = None
        if closure is not None:
            with torch.enable_grad():
                starting_loss = closure(backward=True)

        state = self.state

        flat_grad = self._gather_flat_grad()
        if len(state['xi']) == 0:
            state['xi'] = torch.full([1], self.defaults['xi0'], dtype=flat_grad.dtype, device=flat_grad.device)
        xi = state['xi']

        if len(state['step']) == 0:
            state['step'] = torch.full([1], 0, dtype=flat_grad.dtype, device=flat_grad.device)

        state['step'] += 1

        handle_negative_ratio = True
        if self.defaults['solver'] == 'exact':
            self.zero_hessian()
            self._compute_hessian()

            if self.defaults['average_conv_kernel']:
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None or p.hessian is None:
                            continue

                        if p.dim() == 4:
                            p.hessian = p.hessian.mean(dim=[2, 3], keepdim=True).expand_as(p.hessian).clone()

            nu, param_step = self._solve_subproblem_exact(loss=starting_loss, flat_grad=flat_grad, xi=xi)
            improvement_ratio, gamma_bad = self._improvement_ratio(param_step, starting_loss, flat_grad, closure, nu)
        else:
            raise ValueError('Choose solver between "exact"')

        if norm(param_step) <= self.defaults['grad_tol']:
            print(RuntimeWarning('Return: grad_tol reached'))

        state['improvement_ratio'] = improvement_ratio
        # See [3], Chapter 17 Practicalities for the update rules bellow
        if improvement_ratio >= self.defaults['eta1']:
            if improvement_ratio > self.defaults['eta2']:
                xi.copy_(((self.defaults['alpha1'] * norm(param_step.detach()) ** 3).max(xi)).min(self.max_xi))
        elif handle_negative_ratio and improvement_ratio < 0:

            xi.copy_(torch.min(self.defaults['alpha2'] * norm(param_step.detach()) ** 3,
                               torch.max(torch.tensor(self.defaults['gamma1'],
                                                      device=flat_grad.device), gamma_bad) * xi))
            print('gamma_bad {}'.format(gamma_bad))
        else:
            start_idx = 0
            for param in self.get_params():
                num_els = param.numel()
                curr_upd = param_step[start_idx:start_idx + num_els]
                param.data.add_(-curr_upd.view_as(param), alpha=1)
                start_idx += num_els
            xi.copy_(torch.max(self.defaults['alpha2'] * norm(param_step.detach()) ** 3, self.min_xi))

        return starting_loss
