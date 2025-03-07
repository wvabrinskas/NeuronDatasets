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
  static var randomSeed: UInt64 = 1234
 
  static func randomIn(_ range: ClosedRange<Float>, seed: UInt64 = .random(in: 0...UInt64.max)) -> (num: Float, seed: UInt64) {
    let seedToUse: UInt64 = if Bundle.isRunningTests {
      randomSeed
    } else {
      seed
    }

    var generator = SeededRandomNumberGenerator(seed: seedToUse)
    
    return (Float.random(in: range, using: &generator), seed)
  }
}

// A custom random number generator that uses a seed
// A custom random number generator that uses a seed
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        // XorShift algorithm for pseudorandom number generation
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
