//
//  Extensions.swift
//  NeuronDatasets
//
//  Created by William Vabrinskas on 2/24/25.
//

import Foundation

extension Float {
 
  static func randomIn(_ range: ClosedRange<Float>, seed: UInt64 = .random(in: 0...UInt64.max)) -> (num: Float, seed: UInt64) {
    var generator = SeededRandomNumberGenerator(seed: seed)
    
    return (Float.random(in: range, using: &generator), seed)
  }
}

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
