/**
 * Tests for Skewed Student-t Distribution
 * 
 * Run with: npm test (after adding jest to package.json)
 * Or manually verify by running in Node.js
 */

import { SkewedStudentT } from '../skewed-t';

/**
 * Basic validation tests
 * These can be run manually or with a test runner
 */
export function validateSkewedT() {
  const dist = new SkewedStudentT();
  
  console.log('Testing Skewed Student-t Distribution...\n');
  
  // Test 1: PPF should return values for standard percentiles
  console.log('Test 1: Standard percentiles with nu=5, lambda=-0.3');
  const percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
  const quantiles = dist.ppf(percentiles, [5.0, -0.3]) as number[];
  
  percentiles.forEach((p, i) => {
    console.log(`  P(${p.toFixed(2)}) = ${quantiles[i].toFixed(4)}`);
  });
  
  // Test 2: Median should be close to 0 for symmetric-ish distributions
  console.log('\nTest 2: Median with lambda=0 (symmetric)');
  const medianSymmetric = dist.ppf(0.5, [5.0, 0]) as number;
  console.log(`  Median (lambda=0): ${medianSymmetric.toFixed(6)} (should be ≈ 0)`);
  
  // Test 3: Negative skew test - left tail should be fatter than right
  console.log('\nTest 3: Skewness test (lambda=-0.3)');
  const q01 = dist.ppf(0.01, [5.0, -0.3]) as number;
  const q99 = dist.ppf(0.99, [5.0, -0.3]) as number;
  const median = dist.ppf(0.5, [5.0, -0.3]) as number;
  
  const leftTail = Math.abs(median - q01);
  const rightTail = Math.abs(q99 - median);
  
  console.log(`  1st percentile: ${q01.toFixed(4)}`);
  console.log(`  Median: ${median.toFixed(4)}`);
  console.log(`  99th percentile: ${q99.toFixed(4)}`);
  console.log(`  Left tail width: ${leftTail.toFixed(4)}`);
  console.log(`  Right tail width: ${rightTail.toFixed(4)}`);
  console.log(`  Left/Right ratio: ${(leftTail / rightTail).toFixed(4)} (should be > 1 for negative skew)`);
  
  // Test 4: Sample statistics
  console.log('\nTest 4: Sample statistics (10,000 samples)');
  const samples = dist.random(10000, 5.0, -0.3);
  
  const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const variance = samples.reduce((a, b) => a + Math.pow(b - sampleMean, 2), 0) / (samples.length - 1);
  const sampleStd = Math.sqrt(variance);
  
  // Calculate skewness
  const m3 = samples.reduce((a, b) => a + Math.pow((b - sampleMean) / sampleStd, 3), 0) / samples.length;
  const n = samples.length;
  const skew = (n / ((n - 1) * (n - 2))) * m3 * (n - 1);
  
  console.log(`  Sample mean: ${sampleMean.toFixed(4)} (should be ≈ 0)`);
  console.log(`  Sample std: ${sampleStd.toFixed(4)}`);
  console.log(`  Sample skew: ${skew.toFixed(4)} (should be < 0 for lambda=-0.3)`);
  
  // Test 5: Edge cases
  console.log('\nTest 5: Edge cases');
  try {
    const veryLow = dist.ppf(0.001, [5.0, -0.3]) as number;
    const veryHigh = dist.ppf(0.999, [5.0, -0.3]) as number;
    console.log(`  P(0.001): ${veryLow.toFixed(4)}`);
    console.log(`  P(0.999): ${veryHigh.toFixed(4)}`);
    console.log('  ✓ Edge cases handled');
  } catch (e) {
    console.log('  ✗ Edge case error:', e);
  }
  
  console.log('\nAll tests completed!');
  console.log('\nExpected behavior:');
  console.log('- Median should be close to 0');
  console.log('- Left tail should be wider than right tail (negative skew)');
  console.log('- Sample skewness should be negative when lambda is negative');
  console.log('- No errors or infinite values');
}

// Comparison values from Python (for reference)
// These were generated with:
// from arch.univariate import SkewStudent
// dist = SkewStudent()
// p = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
// print(dist.ppf(p, [5.0, -0.3]))
//
// Expected output (approximate):
// [-3.36, -2.02, -1.43, 0.07, 1.18, 1.57, 2.14]

// Run validation if this file is executed directly
if (require.main === module) {
  validateSkewedT();
}