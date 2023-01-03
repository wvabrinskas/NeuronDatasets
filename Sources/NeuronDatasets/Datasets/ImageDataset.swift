//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/2/23.
//

import Combine
import Foundation
import Neuron
import Logger
#if os(macOS)
import Cocoa
#endif
#if os(iOS)
import UIKit
#endif

/// Creates an RGB dataset from a directory of images. Alpha is removed.
public class ImageDataset: Dataset, Logger {
  public var logLevel: LogLevel = .low
  public var unitDataSize: Neuron.TensorSize
  public var data: DatasetData = ([], []) {
    didSet {
      dataPassthroughSubject.send(data)
    }
  }
  
  public var complete: Bool = false
  public var dataPassthroughSubject = PassthroughSubject<DatasetData, Never>()
  public var overrideLabel: [Float] = []
  
  private let imagesDirectory: String
  private let zeroCentered: Bool
  private let maxCount: Int
  private let validationSplitPercent: Float
  
  /// Initializes an RGB ImageDataset. The data will be an array of Tensor's of depth 3. One for each channel, excluding Alpha
  /// - Parameters:
  ///   - imagesDirectory: The directory of the images to load. All images should be the same size.
  ///   - imageSize: The expected size of the images
  ///   - label: The label to apply to every image.
  ///   - maxCount: Max count to add to the dataset. Could be useful to save memory. Setting it to 0 will use the whole dataset.
  ///   - validationSplitPercent: Number between 0 and 1. The lower the number the more likely it is the image will be added to the training dataset otherwise it'll be added to the validation dataset.
  ///   - zeroCentered: Format image RGB values between -1 and 1. Otherwise it'll be normalized to between 0 and 1.
  public init(imagesDirectory: URL,
              imageSize: Neuron.TensorSize,
              label: [Float],
              maxCount: Int = 0,
              validationSplitPercent: Float = 0,
              zeroCentered: Bool = false) {
    self.unitDataSize = imageSize
    self.imagesDirectory = imagesDirectory.path
    self.overrideLabel = label
    self.zeroCentered = zeroCentered
    self.maxCount = maxCount
    self.validationSplitPercent = validationSplitPercent
  }
  
  public func build() async -> DatasetData {
    readDirectory()
    return self.data
  }
  
  public func build() {
    readDirectory()
  }
  
  private func getImageTensor(for url: String) -> Tensor {
    guard let rawUrl = URL(string: url) else { return Tensor() }
    
    #if os(macOS)
    if let image = NSImage(contentsOf: rawUrl) {
      return image.asSquareRGBTensor(zeroCenter: zeroCentered)
    }
    #elseif os(iOS)
    if let image = UIImage(contentsOfFile: url) {
      return image.asSquareRGBTensor(zeroCenter: zeroCentered)
    }
    #endif
    
    return Tensor()
  }
 
  private func readDirectory() {
    do {
      let contents = try FileManager.default.contentsOfDirectory(atPath: imagesDirectory)
      
      var training: [DatasetModel] = []
      var validation: [DatasetModel] = []
      
      let maximum = maxCount == 0 ? contents.count : maxCount
      
      for index in 0..<maximum {
        let imageUrl = contents[index]
        let path = "file://" + imagesDirectory.appending("/\(imageUrl)")
        let imageData = getImageTensor(for: path)
        let label = Tensor(overrideLabel)
        if Float.random(in: 0...1) >= validationSplitPercent {
          training.append(DatasetModel(data: imageData, label: label))
        } else {
          validation.append(DatasetModel(data: imageData, label: label))
        }
      }
      
      self.data = (training, validation)
    } catch {
      print(error.localizedDescription)
    }
  }
}
