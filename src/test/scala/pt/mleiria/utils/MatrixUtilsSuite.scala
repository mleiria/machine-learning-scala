package pt.mleiria.utils

import breeze.linalg.DenseMatrix

class MatrixUtilsSuite extends munit.FunSuite:

  test("loadXYDatasetFromMatrix"){
    val xy: DenseMatrix[Double] = DenseMatrix((4.0, 5.0, 6.0, 7.0), (8.0, 9.0, 10.0, 11.0), (12.0, 13.0, 14.0, 15.0))
    println(xy)
    val cols = xy.cols
    val y = xy(::, cols - 1)
    println(y)
    val x = xy(::, 0 until cols - 1)
    println(x)
    assert(x.cols == 3)
    assert(x.rows == 3)
    assert(y.length == 3)
    assert(xy(0,3) == y(0))
    assert(xy(1,3) == y(1))
    assert(xy(2,3) == y(2))

  }
