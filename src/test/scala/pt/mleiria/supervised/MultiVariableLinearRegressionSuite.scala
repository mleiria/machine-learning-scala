package pt.mleiria.supervised

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{abs, round}
import pt.mleiria.preprocess.Scaler
import pt.mleiria.utils.{IOUtils, MatrixUtils, roundDoubleScalar, roundDoubleVector}
import spire.implicits.eqOps

class MultiVariableLinearRegressionSuite extends munit.FunSuite:

  test("gradientDescent") {
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

    val (wFinal, bFinal, jHistory) = gd.fit(bIn, alpha, iterations, wIn)

    val bFinalRounded = roundDoubleScalar(bFinal)
    val wFinalRounded = roundDoubleVector(wFinal)

    assert(bFinalRounded == 0.0)

    assert(wFinalRounded(0) == 0.2)
    assert(wFinalRounded(1) == 0.0)
    assert(wFinalRounded(2) == -0.01)
    assert(wFinalRounded(3) == -0.07)

  }

  test("gradientDescentWithScalerAndOneSample") {
    // Load the dataset from csv file
    val xy: DenseMatrix[Double] = IOUtils.readCsvFromResourcesToMatrix("data/houses.txt")
    // Split the data into xTrain and yTrain
    val (xTrain, yTrain) = MatrixUtils.loadXYDatasetFromMatrix(xy)
    val (xTrainNorm, mean, stdev) = Scaler.zScoreNormalizer(xTrain)
    val iterations = 1000
    val alpha = 1.0e-1
    val bIn = 0.0
    val gd = MultiVariableLinearRegression.apply(xTrainNorm, yTrain)
    val (wNorm, bNorm, history) = gd.fit(bIn, alpha, iterations)

    val xHouse = DenseVector[Double](1200.0, 3.0, 1.0, 40.0)
    val xHouseNorm = (xHouse - mean) / stdev
    val xHousePredict = gd.predict(bNorm, wNorm, xHouseNorm)
    assert(roundDoubleScalar(xHousePredict) == 318.94)
  }



