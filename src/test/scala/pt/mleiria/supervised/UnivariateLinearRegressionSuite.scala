package pt.mleiria.supervised

class UnivariateLinearRegressionSuite extends munit.FunSuite:

  import pt.mleiria.supervised.UnivariateLinearRegression

  // Features
  val xTrain: Array[Double] = Array[Double](1.0, 2.0)
  // Target value
  val yTrain: Array[Double] = Array[Double](300.0, 500.0)

  val gd: UnivariateLinearRegression = UnivariateLinearRegression(xTrain, yTrain)

  test("computeCost_test_1"){
    val res = gd.computeCost(0.5, 0.5)
    println(s"Result: $res")
    assert(res == 84475.8125)

  }

  test("computeCost_test_2") {
    val res = gd.computeCost(0.5, 0.0)
    println(s"Result: $res")
    assert(res == 84675.3125)
  }

  test("gradientDescent") {
    val iterations = 10000
    val alpha = 1.0e-2
    val res = gd.fit(0.0, 0.0, alpha, iterations)

    println(s"w,b found by gradient descent: w = ${(math floor res._1 * 1000) / 1000} b =  ${res._2}")
    assert((math floor res._1 * 1000) / 1000 == 199.992)
    assert((math floor res._2 * 1000) / 1000 == 100.011)
  }



