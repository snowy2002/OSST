"""Dyadic interval arithmetic with proved rational transcendental enclosures.

Only Python integer/Fraction arithmetic is used. No float, Decimal, mpmath,
FLINT, NumPy, transcendental library call, or external package is used.
An interval stores integer endpoints divided by D = 2**BITS.
See the companion manuscript's exact-arithmetic appendix for soundness.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import isqrt

BITS = 160
D = 1 << BITS
EXP_DEGREE = 40
LOG_TERMS = 80
ATAN_TERMS = 50


def ceildiv(a, b):
    if b <= 0:
        raise ValueError('positive denominator required')
    return -((-a)//b)


class I:
    __slots__ = ('lo', 'hi')

    def __init__(self, value=0):
        if isinstance(value, I):
            self.lo, self.hi = value.lo, value.hi
        elif isinstance(value, (int, Q)):
            v = Q(value)
            self.lo = v.numerator*D//v.denominator
            self.hi = ceildiv(v.numerator*D, v.denominator)
        else:
            raise TypeError('Only exact int/Fraction or interval inputs accepted')

    @classmethod
    def raw(cls, lo, hi):
        if not isinstance(lo, int) or not isinstance(hi, int) or lo > hi:
            raise ValueError('invalid dyadic interval')
        obj = object.__new__(cls)
        obj.lo, obj.hi = lo, hi
        return obj

    @classmethod
    def enclose(cls, lo, hi):
        lo, hi = Q(lo), Q(hi)
        if lo > hi:
            raise ValueError('unordered endpoints')
        return cls.raw(lo.numerator*D//lo.denominator,
                       ceildiv(hi.numerator*D, hi.denominator))

    def __add__(self, other):
        other = I(other)
        return I.raw(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __neg__(self):
        return I.raw(-self.hi, -self.lo)

    def __sub__(self, other):
        return self+-I(other)

    def __rsub__(self, other):
        return I(other)+-self

    def __mul__(self, other):
        other = I(other)
        products = (self.lo*other.lo, self.lo*other.hi,
                    self.hi*other.lo, self.hi*other.hi)
        return I.raw(min(products)//D, ceildiv(max(products), D))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError('interval contains zero')
        if self.hi < 0:
            return -(-self).reciprocal()
        return I.raw(D*D//self.hi, ceildiv(D*D, self.lo))

    def __truediv__(self, other):
        return self*I(other).reciprocal()

    def __rtruediv__(self, other):
        return I(other)*self.reciprocal()

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError('integer exponent required')
        if exponent < 0:
            return self.reciprocal()**(-exponent)
        result, base = I(1), self
        while exponent:
            if exponent & 1:
                result = result*base
            base = base*base
            exponent >>= 1
        return result

    def __lt__(self, other):
        return self.hi < I(other).lo

    def __gt__(self, other):
        return I(other) < self

    def __le__(self, other):
        return self.hi <= I(other).lo

    def __ge__(self, other):
        return I(other) <= self

    def exact_equal(self, value):
        other = I(value)
        return self.lo == self.hi == other.lo == other.hi

    def sqrt(self):
        if self.lo < 0:
            raise ValueError('negative square-root interval')
        low = isqrt(self.lo*D)
        high = isqrt(self.hi*D)
        if not (low*low <= self.lo*D < (low+1)**2
                and high*high <= self.hi*D < (high+1)**2):
            raise ArithmeticError('integer square-root witness failed')
        if high*high != self.hi*D:
            high += 1
        return I.raw(low, high)

    def exp(self):
        return I.raw(exp_endpoint(self.lo).lo, exp_endpoint(self.hi).hi)

    def upper_fraction(self):
        return Q(self.hi, D)

    def lower_fraction(self):
        return Q(self.lo, D)

    def text(self):
        return dict(lower=str(self.lower_fraction()), upper=str(self.upper_fraction()))


def imin(*xs):
    xs = tuple(I(x) for x in xs)
    return I.raw(min(x.lo for x in xs), min(x.hi for x in xs))


def imax(*xs):
    xs = tuple(I(x) for x in xs)
    return I.raw(max(x.lo for x in xs), max(x.hi for x in xs))


@lru_cache(maxsize=8192)
def exp_endpoint(x):
    """Enclose exp(x/D) using rational Taylor bounds and repeated squaring."""
    if x == 0:
        return I(1)
    if x < 0:
        return exp_endpoint(-x).reciprocal()
    shifts = 0
    while x*8 > (D << shifts):
        shifts += 1
    y = Q(x, D << shifts)  # 0 < y <= 1/8
    term = total = Q(1)
    for k in range(1, EXP_DEGREE+1):
        term = term*y/k
        total += term
    first_omitted = term*y/(EXP_DEGREE+1)
    tail = first_omitted/(1-y/(EXP_DEGREE+2))
    result = I.enclose(total, total+tail)
    for _ in range(shifts):
        result = result*result
    return result


def log_two():
    z = Q(1, 3)
    partial = 2*sum((z**(2*k+1)/(2*k+1) for k in range(LOG_TERMS)), Q())
    remainder = 2*z**(2*LOG_TERMS+1)/((2*LOG_TERMS+1)*(1-z*z))
    return I.enclose(partial, partial+remainder)


def atan_reciprocal(q):
    z = Q(1, q)
    partial = sum(((-1)**k*z**(2*k+1)/(2*k+1)
                   for k in range(ATAN_TERMS)), Q())
    remainder = z**(2*ATAN_TERMS+1)/(2*ATAN_TERMS+1)
    # ATAN_TERMS = 50 is even: the first omitted term is positive.
    return I.enclose(partial, partial+remainder)


def pi_interval():
    return 16*atan_reciprocal(5)-4*atan_reciprocal(239)


def decimal_ceiling(x, places=6):
    """A printable upward decimal rounding, still using integer arithmetic."""
    x = Q(x)
    power = 10**places
    digits = ceildiv(x.numerator*power, x.denominator)
    sign = '-' if digits < 0 else ''
    whole, frac = divmod(abs(digits), power)
    return sign+str(whole)+'.'+str(frac).zfill(places)
