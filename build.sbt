val scala3Version = "3.0.0"

lazy val root = project
  .in(file("."))
  .settings(
    name := "Machine Learning",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,

    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "0.7.29" % Test,
      "org.scalanlp" %% "breeze" % "2.0",
      "org.scalanlp" %% "breeze-natives" % "2.0"
    )
  )
