package pt.mleiria.utils

import breeze.linalg.{DenseMatrix, DenseVector}

object MatrixUtils:

  /**
   * Splits the input Matrix into one matrix without the last column of the input matrix
   * and one vector with the last columns of the input matrix
   * @param xy the input matrix
   * @return Matrix and Vector
   */
  def loadXYDatasetFromMatrix(xy: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
    val cols = xy.cols
    val y = xy(::, cols - 1)
    val x = xy(::, 0 until cols - 1)
    (x, y)
  }
