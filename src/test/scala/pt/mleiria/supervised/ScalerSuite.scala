package pt.mleiria.supervised

import breeze.linalg.DenseMatrix
import pt.mleiria.preprocess.Scaler
import pt.mleiria.utils.{IOUtils, MathUtils, MatrixUtils}

class ScalerSuite extends munit.FunSuite:

  test("scaler.zScore") {
    val xy = IOUtils.readCsvFromResourcesToMatrix("data/houses.txt")
    val x = MatrixUtils.loadXYDatasetFromMatrix(xy)._1
    val (xTransformed, mean, stdev) = Scaler.zScoreNormalizer(x)
    //println(xTransformed)
    println(mean)
    println(stdev)
    assert(MathUtils.roundDoubleScalar(mean(0)) == 1413.71)
    assert(MathUtils.roundDoubleScalar(mean(1)) == 2.71)
    assert(MathUtils.roundDoubleScalar(mean(2)) == 1.38)
    assert(MathUtils.roundDoubleScalar(mean(3)) == 38.65)

    assert(MathUtils.roundDoubleScalar(stdev(0)) == 414.25)
    assert(MathUtils.roundDoubleScalar(stdev(1)) == 0.66)
    assert(MathUtils.roundDoubleScalar(stdev(2)) == 0.49)
    assert(MathUtils.roundDoubleScalar(stdev(3)) == 25.91)

  }
