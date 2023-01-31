package pt.mleiria.utils

import breeze.linalg.{DenseMatrix, DenseVector}

class IOUtilsSuite extends munit.FunSuite:

  test("IOUtils.readCsvFromResourcesToMatrix") {
    val filePath = "data/houses.txt"
    val matrix: DenseMatrix[Double] = IOUtils.readCsvFromResourcesToMatrix(filePath)
    assert(matrix.cols == 5)
    assert(matrix.rows == 100)

  }
