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
public class ImageDataset: BaseDataset, DatasetMergable {
  
  public struct ImageModel {
    var images: URL
    var labels: URL?
    
    public init(images: URL, labels: URL?) {
      self.images = images
      self.labels = labels
    }
  }
  
  public typealias ImageSorting = (URL, URL) -> Bool
  public enum ImageDatasetError: Error, LocalizedError {
    case imageDepthError
    
    public var errorDescription: String? {
      switch self {
      case .imageDepthError:
        return NSLocalizedString("Expected image depth should be equal to the image size depth", comment: "")
      }
    }
  }
  
  public enum ImageDepth: CaseIterable {
    case rgb, rgba, grayScale
    
    public var expectedDepth: Int {
      switch self {
      case .rgb:
        return 3
      case .rgba:
        return 4
      case .grayScale:
        return 1
      }
    }
  }
  
  enum DataType {
    case training, validation
  }
  
  public enum ValidationType {
    case auto(Float)
    case fromUrl(ImageModel)
  }
    
  private let trainingData: ImageModel
  private let zeroCentered: Bool
  private let maxCount: Int
  private let imageDepth: ImageDepth
  private let imageSorting: ImageSorting?
  
  private var autoValidationData: [DatasetModel] = []
  private var validationData: ImageModel? = nil
  private var autoValidation: Bool = false
  @Percentage
  private var validationSplit: Float = 0.0

  /// Initializes an RGB ImageDataset. This call throws an error if the
  /// - Parameters:
  ///   - imagesDirectory: The directory of the images to load. All images should be the same size.
  ///   - labels: The directory of the CSV of labels associated with the images. Should be 1 line with all labels comma separated, a label should be an integer from 1 to the number of classes. Should match the count of the images in the `imagesDirectory`.
  ///   - imageSorting: The sorting to be used on the images directory based on their urls. This is useful if youre trying to match the order of the labels to the order of the images.
  ///   - imageSize: The expected size of the images
  ///   - label: The label to apply to every image.
  ///   - imageDepth: ImageDepth that describes the expected depth of the images.
  ///   - maxCount: Max count to add to the dataset. Could be useful to save memory. Setting it to 0 will use the whole dataset.
  ///   - validationSplitPercent: Number between 0 and 1. The lower the number the more likely it is the image will be added to the training dataset otherwise it'll be added to the validation dataset.
  ///   - zeroCentered: Format image RGB values between -1 and 1. Otherwise it'll be normalized to between 0 and 1.
  public init(trainingData: ImageModel,
              validation: ValidationType,
              imageSorting: ImageSorting? = nil,
              imageSize: CGSize,
              label: [Tensor.Scalar] = [],
              imageDepth: ImageDepth,
              maxCount: Int = 0,
              zeroCentered: Bool = false) {

    self.zeroCentered = zeroCentered
    self.maxCount = maxCount
    self.imageDepth = imageDepth
    self.imageSorting = imageSorting
    self.trainingData = trainingData
    
    switch validation {
    case .auto(let split):
      validationSplit = split
      self.autoValidation = true
    case .fromUrl(let imageData):
      self.validationData = imageData
    }
        
    super.init(unitDataSize: TensorSize(rows: Int(imageSize.width),
                                        columns: Int(imageSize.height),
                                        depth: imageDepth.expectedDepth),
               overrideLabel: label)
  }
  
  public override func build() async -> DatasetData {
    guard complete == false else {
      print("ImageDataset has already been loaded")
      return self.data
    }

    buildData()

    return await super.build()
  }
  
  public override func build() {
    buildData()
    
    super.build()
  }
  
  public func merge(with dataset: ImageDataset) {
    super.merge(with: dataset)
  }
  
  private func buildData() {
    guard complete == false else {
      print("ImageDataset has already been loaded")
      return
    }
    
    let trainingSamples = readDirectory(type: .training)
    let validationSamples = autoValidation ? autoValidationData : readDirectory(type: .validation)
    
    if validationSamples.isEmpty {
      fatalError("Validation set can not be empty. Please check your dataset")
    }

    data = (trainingSamples, validationSamples)
    complete = true
  }
  
  private func getImageTensor(for url: String) -> Tensor {
    guard let rawUrl = URL(string: url) else {
      return Tensor()
    }
    
    #if os(macOS)
    if let image = NSImage(contentsOf: rawUrl) {
      switch imageDepth {
      case .rgb:
        return image.asRGBTensor(zeroCenter: zeroCentered)
      case .rgba:
        return image.asRGBATensor(zeroCenter: zeroCentered)
      case .grayScale:
        return image.asGrayScaleTensor(zeroCenter: zeroCentered)
      }
    }
    #elseif os(iOS)
    if let image = UIImage(contentsOfFile: url) {
      switch imageDepth {
      case .rgb:
        return image.asRGBTensor(zeroCenter: zeroCentered)
      case .rgba:
        return image.asRGBATensor(zeroCenter: zeroCentered)
      case .grayScale:
        return image.asGrayScaleTensor(zeroCenter: zeroCentered)
      }
    }
    #endif
    
    return Tensor()
  }
  
  internal func getLabelsIfNeeded(type: DataType) throws -> [Tensor]? {
    guard let labels = type == .training ? trainingData.labels : type == .validation ? validationData?.labels : nil else { return nil }
    
    let content = try String(contentsOfFile: labels.absoluteString).trimmingCharacters(in: .decimalDigits.inverted)
    let parsedCSV = content.components(separatedBy: ",").compactMap { Tensor.Scalar($0) }
    let maxLabel = parsedCSV.max
    
    var labelsToReturn: [Tensor] = []
    var zeros = [Tensor.Scalar](repeating: 0, count: Int(maxLabel))
    parsedCSV.forEach { val in
      let index = Int(val - 1)
      zeros[index] = 1.0
      labelsToReturn.append(Tensor(zeros))
      zeros[index] = 0.0
    }
    
    return labelsToReturn
  }
  
  private func readDirectory(type: DataType) -> [DatasetModel] {
    guard let url = type == .training ? trainingData.images : type == .validation ? validationData?.images : nil else { return [] }

    do {
      
      var contents = try FileManager.default.contentsOfDirectory(at: url,
                                                                 includingPropertiesForKeys: nil,
                                                                 options: .skipsHiddenFiles)
      
      if let imageSorting {
        contents = contents.sorted(by: imageSorting)
      }
      
      var samples: [DatasetModel] = []
      
      let maximum = maxCount == 0 ? contents.count : maxCount
      
      let labelsFromUrl = try getLabelsIfNeeded(type: type)
      
      var labelsAddedToData: Set<UInt> = []

      for index in 0..<maximum {
        let imageUrl = contents[index]
        let imageData = getImageTensor(for: imageUrl.absoluteString)
        
        precondition(imageData.shape == unitDataSize.asArray)
        
        let label = labelsFromUrl?[index] ?? Tensor(overrideLabel)
        
        if labelsFromUrl?[safe: index] == nil {
          fatalError("Label is missing for: \(imageUrl)")
        }
        
        let sample = DatasetModel(data: imageData, label: label)
              
        if autoValidation, type != .validation, overrideLabel.isEmpty {
          let labelValue = label.value.flatten().indexOfMax.0
          
          if Float.randomIn(0...1, seed: randomizationSeed).num < validationSplit,
             labelsAddedToData.contains(labelValue) { // only take a validation sample if there's at least one in the training data already
            
            autoValidationData.append(sample) // appends sample from the training data to the validation set
          } else {
            labelsAddedToData.insert(labelValue)
            samples.append(sample)
          }
        } else {
          samples.append(sample)
        }
       
      }
      
      return samples
    } catch {
      print(error.localizedDescription)
      return []
    }
  }
}
