package utils

import breeze.linalg.DenseVector

object MathUtils:

  /**
   *
   * @param d a Double to round up to 2 decimal places
   * @return the rounded Double
   */
  def roundDoubleScalar(d: Double): Double = {
    BigDecimal(d).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
  }
  /**
   *
   * @param v a DenseVector[Double] to round up it's elements to 2 decimal places
   * @return the DenseVector[Double] rounded
   */
  def roundDoubleVector(v: DenseVector[Double]): DenseVector[Double] = {
    v.map(elem => roundDoubleScalar(elem))
  }


