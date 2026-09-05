//
//  TokenizableDataset.swift
//  NeuronDatasets
//
//  Created by William Vabrinskas on 1/9/26.
//

import Neuron
import Foundation

open class ProxyTokenizableDataset: BaseDataset, TokenizingDataset {
  public typealias Tokenizer = BPETokenizer
  
  public var bosTokenId: Int {
    tokenDataset.bosTokenId
  }

  public var padTokenId: Int {
    tokenDataset.padTokenId
  }
  
  public var tokenizer: BPETokenizer {
    tokenDataset.tokenizer
  }

  public var vocabSize: Int {
    tokenDataset.vocabSize
  }
  
  public var eosTokenId: Int {
    tokenDataset.eosTokenId
  }
  
  public var controlTokenIds: Set<Int> {
    tokenDataset.controlTokenIds
  }

  private let tokenDataset: TokenizableDataset
  
  public static func build(url: URL) -> Self {
    Self.init(tokenizer: Tokenizer.import(url), unitDataSize: .init())
  }
  
  public static func build(data: Data) -> Self {
    Self.init(tokenizer: Tokenizer.import(data), unitDataSize: .init())
  }
  
  public required init(tokenizer: Tokenizer,
                       unitDataSize: Neuron.TensorSize,
                       overrideLabel: [Tensor.Scalar] = []) {
    self.tokenDataset = TokenizableDataset(tokenizer: tokenizer)
    
    super.init(unitDataSize: unitDataSize, overrideLabel: overrideLabel)
  }
  
  public func tokenize(_ items: Item) -> Neuron.Tensor {
    tokenDataset.tokenize(items)
  }
  
  public func tokenize(_ item: Item, paddedTo length: Int, appendingEnd: Bool) -> Neuron.Tensor {
    tokenDataset.tokenize(item, paddedTo: length, appendingEnd: appendingEnd)
  }
  
  public func item(for data: Neuron.Tensor) -> Item {
    tokenDataset.item(for: data)
  }
  
  public func export(name: String?, overrite: Bool, compress: Bool) -> URL? {
    tokenDataset.export(name: name, overrite: overrite, compress: compress)
  }
  
  public func tokenCount(for item: String, addingBoundaryTokens: Bool = true) -> Int {
    tokenDataset.tokenCount(for: item, addingBoundaryTokens: addingBoundaryTokens)
  }
  
  public func nextTokenPair(for item: String, sequenceLength: Int, addingBoundaryTokens: Bool = true) -> (data: Neuron.Tensor, label: Neuron.Tensor) {
    tokenDataset.nextTokenPair(for: item, sequenceLength: sequenceLength, addingBoundaryTokens: addingBoundaryTokens)
  }
  
  public func sequenceLength(for items: [String], cappedAt cap: Int?, addingBoundaryTokens: Bool = true) -> Int {
    tokenDataset.sequenceLength(for: items, cappedAt: cap, addingBoundaryTokens: addingBoundaryTokens)
  }
  
  
}
