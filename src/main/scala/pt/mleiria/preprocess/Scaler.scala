package pt.mleiria.preprocess

import breeze.linalg._
import breeze.stats.meanAndVariance

object Scaler {

  /**
   *
   * @param x Matrix to normalize (x - mean) / stdev
   * @return (DenseMatrix normalized, DenseVector mean, DenseVector stdev)
   */
  def zScoreNormalizer(x: DenseMatrix[Double]):
  (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {

    // Number of examples
    val m: Int = x.rows
    // Number of features
    val n: Int = x.cols
    // Each column will have a mean and stdv
    val mean = DenseVector.zeros[Double](n)
    val stdev = DenseVector.zeros[Double](n)

    def loop(numCol: Int): Unit = {
      if (numCol < n) {
        val colMeanAndStdv = meanAndVariance(x(::, numCol))
        mean(numCol) = colMeanAndStdv.mean
        stdev(numCol) = colMeanAndStdv.stdDev
        loop(numCol + 1)
      }
    }

    loop(0)
    // In breeze, vectors are column vectors
    // Broadcast columns
    val xMean = (x(*, ::) - mean)
    val xNorm = xMean(*, ::) / stdev

    (xNorm, mean, stdev)
  }
}
