//
//  Extensions.swift
//  NeuronDatasets
//
//  Created by William Vabrinskas on 2/24/25.
//

import Foundation

extension Bundle {
  static var isRunningTests : Bool {
    get {
      return NSClassFromString("XCTest") != nil
    }
  }
}


extension Float {
  static var randomSeed: Float = 0.1

  static func randomIn(_ range: ClosedRange<Float>) -> Float {
    if Bundle.isRunningTests {
      return randomSeed
    } else {
      return Float.random(in: range)
    }
  }
}
