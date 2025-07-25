// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NeuronDatasets",
    platforms: [ .iOS(.v14),
                 .tvOS(.v14),
                 .watchOS(.v7),
                 .macOS(.v14)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "NeuronDatasets",
            targets: ["NeuronDatasets"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
      .package(url: "https://github.com/wvabrinskas/Neuron.git", from: "2.0.19"),
      //.package(path: "../Neuron"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "NeuronDatasets",
            dependencies: ["Neuron"],
            resources: [ .process("Resources") ]),
        .testTarget(
            name: "NeuronDatasetsTests",
            dependencies: ["NeuronDatasets"]),
    ]
)
