package pt.mleiria.supervised

import breeze.linalg.DenseMatrix
import pt.mleiria.preprocess.Scaler
import pt.mleiria.utils.{IOUtils, MatrixUtils, roundDoubleScalar}

class ScalerSuite extends munit.FunSuite:

  test("scaler.zScore") {
    val xy = IOUtils.readCsvFromResourcesToMatrix("data/houses.txt")
    val x = MatrixUtils.loadXYDatasetFromMatrix(xy)._1
    val (xTransformed, mean, stdev) = Scaler.zScoreNormalizer(x)
    //println(xTransformed)
    println(mean)
    println(stdev)
    assert(roundDoubleScalar(mean(0)) == 1413.71)
    assert(roundDoubleScalar(mean(1)) == 2.71)
    assert(roundDoubleScalar(mean(2)) == 1.38)
    assert(roundDoubleScalar(mean(3)) == 38.65)

    assert(roundDoubleScalar(stdev(0)) == 414.25)
    assert(roundDoubleScalar(stdev(1)) == 0.66)
    assert(roundDoubleScalar(stdev(2)) == 0.49)
    assert(roundDoubleScalar(stdev(3)) == 25.91)

  }
