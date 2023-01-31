package pt.mleiria

import breeze.linalg.*

class BreezeSuite extends munit.FunSuite:

  test("beeze.broadcat.matrix.with.vector.cols") {
    val dm = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val dv = DenseVector(3.0, 4.0)
    println(dm)
    println(dv)
    val res = dm(::, *) + dv
    println(res)
    assert(res == DenseMatrix((4.0, 5.0, 6.0), (8.0, 9.0, 10.0)))
  }

  test("beeze.broadcat.matrix.with.vector.rows") {
    val dm = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val dv = DenseVector(3.0, 4.0, 5.0)
    println(dm)
    println(dv)
    val res = dm(*, ::) + dv
    println(res)
    assert(res == DenseMatrix((4.0, 6.0, 8.0), (7.0, 9.0, 11.0)))
  }

  test("beeze.broadcat.matrix.with.scalar") {
    val dm = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val s = 1.0
    println(dm)
    val res = dm(::, *) + s
    println(res)
    assert(res == DenseMatrix((2.0, 3.0, 4.0), (5.0, 6.0, 7.0)))
  }
