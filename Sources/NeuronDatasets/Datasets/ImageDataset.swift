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
    
  private let imagesDirectory: String
  private let zeroCentered: Bool
  private let maxCount: Int
  private let validationSplitPercent: Tensor.Scalar
  private let imageDepth: ImageDepth
  private let labels: URL?
  private let imageSorting: ImageSorting?
  
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
  public init(imagesDirectory: URL,
              labels: URL? = nil,
              imageSorting: ImageSorting? = nil,
              imageSize: CGSize,
              label: [Tensor.Scalar] = [],
              imageDepth: ImageDepth,
              maxCount: Int = 0,
              validationSplitPercent: Tensor.Scalar = 0,
              zeroCentered: Bool = false) {

    self.imagesDirectory = imagesDirectory.path
    self.zeroCentered = zeroCentered
    self.maxCount = maxCount
    self.validationSplitPercent = validationSplitPercent
    self.imageDepth = imageDepth
    self.labels = labels
    self.imageSorting = imageSorting
    
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
    
    readDirectory()
    return await super.build()
  }
  
  public override func build() {
    readDirectory()
    super.build()
  }
  
  public func merge(with dataset: ImageDataset) {
    super.merge(with: dataset)
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
  
  internal func getLabelsIfNeeded() throws -> [Tensor]? {
    guard let labels else { return nil }
    
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
 
  private func readDirectory() {
    guard complete == false else {
      print("ImageDataset has already been loaded")
      return
    }
    
    do {
      guard let url = URL(string: imagesDirectory) else { return }
      
      var contents = try FileManager.default.contentsOfDirectory(at: url,
                                                                 includingPropertiesForKeys: nil,
                                                                 options: .skipsHiddenFiles)
      
      if let imageSorting {
        contents = contents.sorted(by: imageSorting)
      }
      
      var training: [DatasetModel] = []
      var validation: [DatasetModel] = []
      
      let maximum = maxCount == 0 ? contents.count : maxCount
      
      let labelsFromUrl = try getLabelsIfNeeded()
      
      for index in 0..<maximum {
        let imageUrl = contents[index]
        let imageData = getImageTensor(for: imageUrl.absoluteString)
        
        precondition(imageData.shape == unitDataSize.asArray)
        
        let label = labelsFromUrl?[index] ?? Tensor(overrideLabel)
        if self.labels != nil, labelsFromUrl?[safe: index] == nil {
          fatalError("Label is missing for: \(imageUrl)")
        }
        if Tensor.Scalar.random(in: 0...1) >= validationSplitPercent {
          training.append(DatasetModel(data: imageData, label: label))
        } else {
          validation.append(DatasetModel(data: imageData, label: label))
        }
      }
      
      self.data = (training, validation)
      complete = true
    } catch {
      print(error.localizedDescription)
    }
  }
}
