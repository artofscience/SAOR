Functions
==============

.. math:: 
    \newcommand{\bx}{\mathbf{x}}
    \newcommand{\by}{\mathbf{y}}
    \newcommand{\x}{\textrm{x}}
    \newcommand{\y}{\textrm{y}}
    \newcommand{\d}{\textrm{d}}
    \newcommand{\bc}{\mathbf{c}}
    \newcommand{\bf}{\mathbf{f}}

The approximate response functions take the form

.. math::
    \begin{equation}
        f[\bx] = \d\bf_k \cdot \left( \d\by_k \right)^{-1} \cdot \left( \by - \by_k \right) + {\textstyle\frac{1}{2}} \bc_k \cdot \left( \bx - \bx_k \right)^2
    \end{equation}
wherein

.. math::
    \d\bf_k = \left.\frac{\d f}{\d \bx}\right|_{\bx_k}
and

.. math::
    \d\by_k = \left. \left[ \frac{\d \y_1}{\d \x_1}, \frac{\d \y_2}{\d \x_2}, \ldots, \frac{\d \y_n}{\d \x_n} \right] \right|_{\bx_k}

A subscripted boldsymbol (an array) indicates an iteration number (:math:`\bx_k`), whereas an element in such an array is denoted by normal text, with the subscript then denoting the index (:math:`\textrm{x}_i`).

..
    .. math::
   \begin{array}{ll}
     \mbox{minimize} & \frac{1}{2} x^T P x + q^T x \\
     \mbox{subject to} & l \leq A x \leq u
   \end{array}

.. automodule:: sao.function
   :members:
   :undoc-members:
   :show-inheritance:
