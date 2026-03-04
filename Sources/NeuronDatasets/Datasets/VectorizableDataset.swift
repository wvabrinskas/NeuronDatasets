//
//  VectorizableDataset.swift
//  NeuronDatasets
//
//  Created by William Vabrinskas on 1/9/26.
//

import Neuron
import Foundation

open class VectorizableDataset: BaseDataset, VectorizingDataset {

  public let vectorizer: Vectorizer
  
  public var vocabSize: Int = 0
  
  public required init(vectorizer: Vectorizer = .init(),
                       unitDataSize: Neuron.TensorSize,
                       overrideLabel: [Tensor.Scalar] = []) {
    self.vectorizer = vectorizer
    self.vocabSize = vectorizer.vector.count
    
    super.init(unitDataSize: unitDataSize, overrideLabel: overrideLabel)
  }

  public static func build(url: URL) -> Self {
    Self.init(vectorizer: Vectorizer.import(url), unitDataSize: .init())
  }

  public static func build(data: Data) -> Self {
    return Self.init(vectorizer: Vectorizer.import(data), unitDataSize: .init())
  }
  
  public func oneHot(_ items: [String]) -> Tensor {
    vectorizer.oneHot(items)
  }
  
  public func vectorize(_ items: [String]) -> Tensor {
    Tensor(items.map { [[Tensor.Scalar(vectorizer.vector[$0, default: 0])]] })
  }
  
  /// Decodes model output back into vector items.
  ///
  /// - Parameters:
  ///   - data: Tensor to decode.
  ///   - oneHot: Whether `data` uses one-hot encoding.
  /// - Returns: Decoded vector items.
  public func getWord(for data: Tensor, oneHot: Bool) -> [String] {
    if oneHot == false {
      let intArray = data.storage.map { Int($0) }
      return vectorizer.unvectorize(intArray)
    } else {
      return vectorizer.unvectorizeOneHot(data)
    }
  }
  
  public func export(name: String?, overrite: Bool, compress: Bool) -> URL?  {
    vectorizer.export(name: name, overrite: overrite, compress: compress)
  }
}
