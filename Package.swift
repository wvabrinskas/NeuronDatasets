// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NeuronDatasets",
    platforms: [ .iOS(.v13),
                 .tvOS(.v13),
                 .watchOS(.v6),
                 .macOS(.v11)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "NeuronDatasets",
            targets: ["NeuronDatasets"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
         //.package(url: "https://github.com/wvabrinskas/Neuron.git", from: "2.0.4")
      .package(url: "../Neuron", branch: "gpu_support_v1")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "NeuronDatasets",
            dependencies: ["Neuron"],
            exclude: [ "bin" ],
            resources: [ .process("Resources") ]),
        .testTarget(
            name: "NeuronDatasetsTests",
            dependencies: ["NeuronDatasets"]),
    ]
)
