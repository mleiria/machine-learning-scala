package pt.mleiria.supervised

import breeze.linalg.{DenseMatrix, DenseVector}
import utils.MathUtils

class MultiVariableLinearRegressionSuite extends munit.FunSuite:
  // Dados de treino do nosso modelo
  val xTrain: DenseMatrix[Double] = DenseMatrix((2104.0, 5.0, 1.0, 45.0), (1416.0, 3.0, 2.0, 40.0), (852.0, 2.0, 1.0, 35.0))
  val yTrain: DenseVector[Double] = DenseVector(460.0, 232.0, 178.0)

  val m: Int = xTrain.rows
  val n: Int = xTrain.cols
  // Initialize parameters
  val wIn: DenseVector[Double] = DenseVector.zeros[Double](n)
  val bIn = 0.0
  // some gradient descent settings
  val iterations = 1000
  val alpha = 5.0e-7
  // Run gradient descent
  val gd: MultiVariableLinearRegression = MultiVariableLinearRegression(xTrain, yTrain)



  test("gradientDescent") {
    val (wFinal, bFinal, jHistory) = gd.fit(wIn, bIn, alpha, iterations)

    val bFinalRounded = MathUtils.roundDoubleScalar(bFinal)
    val wFinalRounded = MathUtils.roundDoubleVector(wFinal)

    assert(bFinalRounded == 0.0)

    assert(wFinalRounded(0) == 0.2)
    assert(wFinalRounded(1) == 0.0)
    assert(wFinalRounded(2) == -0.01)
    assert(wFinalRounded(3) == -0.07)

  }



