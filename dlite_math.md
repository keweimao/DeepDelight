
# DLITE Derivative 

## LIT derivative

Given the following g(x) function (for LIT) in LaTeX:  

```latex
g(x) = \int_{c}^{x} - \log(x) \, dx
```

Follow the Theorem of Calculus, the derivative is the integrand: 

```latex
g'(x) = - \log(x)
```

## dH derivative

Given the following h(x) function (for entropy discount) in Latex: 

```latex
h(x) = \abs(x-c) \frac{\int -x \log{x} \, dx}{\int x \, dx}
```

Or: 

```latex
h(x) = \int_{c}^{x} dx \frac{\int_{c}^{x} -x \log{x} \, dx}{\int_{c}^{x} x \, dx}
```

Break down the given function \( h(x) \):

```latex
\[ h(x) = \int_{c}^{x} dx \frac{\int_{c}^{x} -x \log{x} \, dx}{\int_{c}^{x} x \, dx} \]
```

<!-- Given the integral with respect to \( x \) is mentioned twice in the expression (one outside the fraction and another inside), this formulation seems unusual and may contain a typo. Nonetheless, let's try to differentiate the function as it's presented. -->

We'll label parts of the function for clarity:

```latex
1. \( u(x) = \int_{c}^{x} dx \) which is simply \( u(x) = x - c \)
2. \( v(x) = \int_{c}^{x} -x \log{x} \, dx \)
3. \( w(x) = \int_{c}^{x} x \, dx \)
```

So, 

```latex
\[ h(x) = u(x) \cdot \frac{v(x)}{w(x)} \]
```

We want to find \( h'(x) \), and for that, we'll apply both the product rule and the quotient rule for differentiation.

Using the product rule combined with the quotient rule:

```latex
\[ h'(x) = u'(x) \cdot \frac{v(x)}{w(x)} + u(x) \cdot \frac{v'(x)w(x) - v(x)w'(x)}{w^2(x)} \]
```

Differentiating each component:

1. \( u'(x) = 1 \) since \( \frac{d}{dx} (x - c) = 1 \)
2. \( v'(x) = -x \log(x) \) due to the Fundamental Theorem of Calculus
3. \( w'(x) = x \) similarly due to the Fundamental Theorem of Calculus

Plugging these values into our equation for \( h'(x) \):

```latex
\[ h'(x) = 1 \cdot \frac{-x \log(x)}{x} + (x - c) \cdot \frac{-x \log(x) \cdot x - (-x \log(x) \cdot x)}{x^2} \]
```

This simplifies to:

```latex
\[ h'(x) = -\log(x) + (x - c) \cdot 0 \]
\[ h'(x) = -\log(x) \]
```

So, the derivative \( h'(x) \) is:

```latex
\[ h'(x) = -\log(x) \]
```

## DLITE derivative

f(x) = g(x) - h(x)

Then $f'(x) = g'(x) - h'(x) = 0$

How could this be? 

