# phys-autodiff
---

## 🟦 **Slide 1 — The Problem: Why “Double Backward” Exists**

**Title:** *Physics-Informed NeRF → Derivatives of Derivatives*

**Equation (Chu et al. 2022):**
[
L_{\text{phys}}
=|\partial_t\rho+\mathbf u!\cdot!\nabla\rho-\kappa\nabla^2\rho|^2
+|\partial_t\mathbf u+\mathbf u!\cdot!\nabla\mathbf u-\nu\nabla^2\mathbf u|^2
+\lambda|\nabla!\cdot!\mathbf u|^2
]

**Network output:**
(\rho,\mathbf u=f_\theta(x,t))

**During training:**
[
\frac{\partial L_{\text{phys}}}{\partial\theta}
=\frac{\partial L}{\partial(\partial\rho)}\frac{\partial(\partial\rho)}{\partial\theta}
\Rightarrow \text{second-order autodiff = double backward.}
]

**Bottom tagline:**
→ High memory     → Slow training     → Gradient noise & instability

---

## 🟦 **Slide 2 — The Idea: Replace Residuals with a Differentiable Solver**

**Title:** *Differentiate the Step, not the Residual*

**Treuille et al. 2003 (Equation 7 & Section 4):**
[
\min_{u_{\rm ctrl}};J
=|\rho(T)-\rho_{\text{target}}|^2
\quad
\text{s.t.}\quad
q_{t+1}=S(q_t,u_{\rm ctrl})
]
→ The solver (S) encodes **advection + diffusion + projection**

**Our training loss:**
[
L_{\text{step}}
=|q_{t+\Delta t}^{\text{net}}-S(q_t^{\text{net}})|^2
]

**Backward path:**
[
\frac{\partial L_{\text{step}}}{\partial\theta}
=\frac{\partial L}{\partial S}\frac{\partial S}{\partial\theta}
\quad\text{(single chain, no second derivative).}
]

**Visual cue:**
Left → “PDE residual graph (two red arrows)”
Right → “Solver block (single arrow)”

---

## 🟦 **Slide 3 — The Replacement: From Chu’s Losses to Operator Consistency**

| Original (Chu et al. 2022)                                                                                                      | Our Operator Form (inspired by Treuille 2003)                                                             |
| :------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------- |
| (L_{D\sigma/Dt}=|\partial_t\sigma+\mathbf u!\cdot!\nabla\sigma|^2)                                                              | (L_{\text{adv}}=|\sigma_{t+\Delta t}-\text{Advect}[\sigma_t,\mathbf u_t]|^2)                              |
| (L_{\text{NSE}}=|\partial_t\mathbf u+\mathbf u!\cdot!\nabla\mathbf u-\nu\nabla^2\mathbf u|^2+\lambda|\nabla!\cdot!\mathbf u|^2) | (L_{\text{NSstep}}=|\mathbf u_{t+\Delta t}-\text{Project}(\text{Diffuse}(\text{Advect}[\mathbf u_t]))|^2) |

**Key difference:**
[
\text{PDE residual (loss on derivatives)}
;\Longrightarrow;
\text{Discrete evolution (loss on solver outputs)}.
]

**Result:**
[
\text{Physics kept, double backward removed.}
]

---
