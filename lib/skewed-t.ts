/**
 * Implementation of Hansen's Skewed Student-t distribution
 * Reference: Hansen (1994) "Autoregressive Conditional Density Estimation"
 * 
 * This is used by the ARCH library in Python and needs to be carefully ported
 * to ensure the same statistical properties are maintained.
 */

/**
 * Approximation of the gamma function using Lanczos approximation
 */
function gamma(z: number): number {
  // Lanczos approximation coefficients
  const g = 7;
  const coef = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];

  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  }

  z -= 1;
  let x = coef[0];
  for (let i = 1; i < g + 2; i++) {
    x += coef[i] / (z + i);
  }

  const t = z + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

/**
 * Error function approximation
 */
function erf(x: number): number {
  // Abramowitz and Stegun approximation
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);

  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return sign * y;
}

/**
 * Standard normal CDF
 */
function normalCDF(x: number): number {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

/**
 * Student's t CDF approximation
 */
function studentTCDF(x: number, nu: number): number {
  if (nu <= 0) throw new Error('Degrees of freedom must be positive');
  
  // For large degrees of freedom, approximate with normal
  if (nu > 1000) {
    return normalCDF(x);
  }

  // Use beta function relationship
  const t = x;
  const a = nu / 2;
  const b = 0.5;
  
  if (t === 0) return 0.5;
  
  const z = nu / (nu + t * t);
  
  // Incomplete beta function approximation
  let betaVal: number;
  if (z < (a + 1) / (a + b + 2)) {
    betaVal = incompleteBeta(z, a, b) / beta(a, b);
  } else {
    betaVal = 1 - incompleteBeta(1 - z, b, a) / beta(b, a);
  }
  
  if (t > 0) {
    return 1 - 0.5 * betaVal;
  } else {
    return 0.5 * betaVal;
  }
}

/**
 * Beta function
 */
function beta(a: number, b: number): number {
  return (gamma(a) * gamma(b)) / gamma(a + b);
}

/**
 * Incomplete beta function (approximation)
 */
function incompleteBeta(x: number, a: number, b: number): number {
  if (x === 0) return 0;
  if (x === 1) return 1;
  
  // Use continued fraction expansion
  const lbeta = Math.log(beta(a, b));
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  
  const f = continuedFraction(x, a, b);
  return front * f;
}

/**
 * Continued fraction for incomplete beta
 */
function continuedFraction(x: number, a: number, b: number, maxIter: number = 200): number {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;
  
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  let h = d;
  
  for (let m = 1; m <= maxIter; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    h *= d * c;
    
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const del = d * c;
    h *= del;
    
    if (Math.abs(del - 1) < 1e-10) break;
  }
  
  return h;
}

/**
 * Inverse of standard normal CDF (PPF/quantile function)
 * Using Beasley-Springer-Moro algorithm
 */
function normalPPF(p: number): number {
  if (p <= 0 || p >= 1) {
    throw new Error('Probability must be between 0 and 1');
  }

  const a = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00
  ];

  const b = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01
  ];

  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00
  ];

  const d = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number, r: number;

  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
}

/**
 * Student's t PPF (inverse CDF)
 * Using Newton-Raphson method
 */
function studentTPPF(p: number, nu: number): number {
  if (p <= 0 || p >= 1) {
    throw new Error('Probability must be between 0 and 1');
  }
  if (nu <= 0) {
    throw new Error('Degrees of freedom must be positive');
  }

  // For large nu, approximate with normal
  if (nu > 1000) {
    return normalPPF(p);
  }

  // Initial guess from normal approximation
  let x = normalPPF(p);
  
  // Newton-Raphson refinement
  for (let i = 0; i < 10; i++) {
    const cdf = studentTCDF(x, nu);
    const pdf = studentTPDF(x, nu);
    
    if (Math.abs(pdf) < 1e-30) break;
    
    const dx = (cdf - p) / pdf;
    x -= dx;
    
    if (Math.abs(dx) < 1e-10) break;
  }
  
  return x;
}

/**
 * Student's t PDF
 */
function studentTPDF(x: number, nu: number): number {
  const coef = gamma((nu + 1) / 2) / (Math.sqrt(nu * Math.PI) * gamma(nu / 2));
  return coef * Math.pow(1 + (x * x) / nu, -(nu + 1) / 2);
}

/**
 * Hansen's Skewed Student-t distribution
 * Parameters:
 * - nu: degrees of freedom (controls tail fatness, lower = fatter tails)
 * - lambda: asymmetry parameter (negative = left skew, positive = right skew)
 */
export class SkewedStudentT {
  /**
   * Calculate the PPF (Percent Point Function / Quantile Function) 
   * for Hansen's skewed Student-t distribution
   * 
   * This matches the behavior of arch.univariate.SkewStudent().ppf()
   */
  ppf(p: number | number[], params: [number, number]): number | number[] {
    const [nu, lambda] = params;
    
    if (Array.isArray(p)) {
      return p.map(pi => this.ppfSingle(pi, nu, lambda));
    }
    return this.ppfSingle(p, nu, lambda);
  }

  private ppfSingle(p: number, nu: number, lambda: number): number {
    if (p <= 0 || p >= 1) {
      if (p === 0) return -Infinity;
      if (p === 1) return Infinity;
      throw new Error('Probability must be between 0 and 1');
    }

    // Hansen's parameterization constants
    const a = 4 * lambda * ((nu - 2) / (nu - 1));
    const b = Math.sqrt(1 + 3 * lambda * lambda - a * a);
    
    // Determine which tail we're in
    const c = (1 + lambda) / 2;
    
    let z: number;
    if (p < c) {
      // Left tail
      const pAdj = p / c;
      const tVal = studentTPPF(pAdj, nu);
      z = (tVal * b - a) / (1 + lambda);
    } else {
      // Right tail
      const pAdj = (p - c) / (1 - c);
      const tVal = studentTPPF(pAdj, nu);
      z = (tVal * b - a) / (1 - lambda);
    }
    
    return z;
  }

  /**
   * Generate random samples from the skewed Student-t distribution
   */
  random(size: number, nu: number, lambda: number): number[] {
    const samples: number[] = [];
    for (let i = 0; i < size; i++) {
      const p = Math.random();
      samples.push(this.ppfSingle(p, nu, lambda));
    }
    return samples;
  }
}

export default SkewedStudentT;