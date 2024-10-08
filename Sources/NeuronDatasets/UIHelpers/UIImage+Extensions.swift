//
//  UIImage+Extensions.swift
//  GanTester
//
//  Created by William Vabrinskas on 3/22/22.
//

import Neuron
#if os(iOS)
import Foundation
import UIKit
import NumSwift

public extension Tensor.Scalar {
  var bytes: [UInt8] {
    withUnsafeBytes(of: self, Array.init)
  }
}

#if arch(arm64)
public extension Float16 {
  var bytes: [UInt8] {
    withUnsafeBytes(of: self, Array.init)
  }
}
#endif

public extension UIImage {
  
  struct PixelData {
    var a: UInt8
    var r: UInt8
    var g: UInt8
    var b: UInt8
  }
  
  func asGrayScaleTensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = self.cgImage?.dataProvider?.data else { return Tensor() }
    
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var grayArray: [Tensor.Scalar] = []

    for y in 0..<Int(self.size.height) {
      for x in 0..<Int(self.size.width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        let r = Tensor.Scalar(data[pixelInfo])
        let g = Tensor.Scalar(data[pixelInfo + 1])
        let b = Tensor.Scalar(data[pixelInfo + 2])
        
        var gray = (r + g + b) / 3

        if zeroCenter {
          gray = (gray - 127.5) / 127.5
        } else {
          gray = gray / 255.0
        }
        
        grayArray.append(gray)
      }
    }
    
    return Tensor([grayArray.reshape(columns: Int(self.size.width))])
  }
  
  func asRGBATensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = self.cgImage?.dataProvider?.data else { return Tensor() }
    
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var rArray: [Tensor.Scalar] = []
    var gArray: [Tensor.Scalar] = []
    var bArray: [Tensor.Scalar] = []
    var aArray: [Tensor.Scalar] = []

    for y in 0..<Int(self.size.height) {
      for x in 0..<Int(self.size.width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        var r = Tensor.Scalar(data[pixelInfo])
        var g = Tensor.Scalar(data[pixelInfo + 1])
        var b = Tensor.Scalar(data[pixelInfo + 2])
        var a = Tensor.Scalar(data[pixelInfo + 3])

        if zeroCenter {
          r = (r - 127.5) / 127.5
          g = (g - 127.5) / 127.5
          b = (b - 127.5) / 127.5
          a = (a - 127.5) / 127.5
        } else {
          r = r / 255.0
          g = g / 255.0
          b = b / 255.0
          a = a / 255.0
        }
        
        rArray.append(r)
        gArray.append(g)
        bArray.append(b)
        aArray.append(a)
      }
    }
    
    return Tensor([rArray.reshape(columns: Int(self.size.width)),
                   gArray.reshape(columns: Int(self.size.width)),
                   bArray.reshape(columns: Int(self.size.width)),
                   aArray.reshape(columns: Int(self.size.width))])
  }
  
  func asRGBTensor(zeroCenter: Bool = false) -> Tensor {
    guard let pixelData = self.cgImage?.dataProvider?.data else { return Tensor() }
    
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    var rArray: [Tensor.Scalar] = []
    var gArray: [Tensor.Scalar] = []
    var bArray: [Tensor.Scalar] = []
    
    for y in 0..<Int(self.size.height) {
      for x in 0..<Int(self.size.width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        var r = Tensor.Scalar(data[pixelInfo])
        var g = Tensor.Scalar(data[pixelInfo + 1])
        var b = Tensor.Scalar(data[pixelInfo + 2])
        
        if zeroCenter {
          r = (r - 127.5) / 127.5
          g = (g - 127.5) / 127.5
          b = (b - 127.5) / 127.5
        } else {
          r = r / 255.0
          g = g / 255.0
          b = b / 255.0
        }
        
        rArray.append(r)
        gArray.append(g)
        bArray.append(b)
      }
    }
    
    return Tensor([rArray.reshape(columns: Int(self.size.width)),
                   gArray.reshape(columns: Int(self.size.width)),
                   bArray.reshape(columns: Int(self.size.width))])
  }
  
  func asPixels() -> [PixelData] {
    var returnPixels = [PixelData]()
    
    guard let pixelData = self.cgImage?.dataProvider?.data else { return [] }
    
    let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
    
    for y in 0..<Int(self.size.height) {
      for x in 0..<Int(self.size.width) {
        let pos = CGPoint(x: x, y: y)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y) * 4) + Int(pos.x) * 4)
        
        let r = data[pixelInfo]
        let g = data[pixelInfo + 1]
        let b = data[pixelInfo + 2]
        let a = data[pixelInfo + 3]
        returnPixels.append(PixelData(a: a, r: r, g: g, b: b))
      }
    }
    return returnPixels
  }
  
  static func from(_ pixels: [Tensor.Scalar], size: (Int, Int)) -> UIImage? {
    let data: [UInt8] = pixels.map { UInt8(ceil(Double($0) * 255)) }
    
    guard data.count >= 8 else {
      print("data too small")
      return nil
    }
    
    let width  = size.0
    let height = size.1
    
    let colorSpace = CGColorSpaceCreateDeviceGray()
    
    guard data.count >= width * height,
          let context = CGContext(data: nil,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: width,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.alphaOnly.rawValue),
          let buffer = context.data?.bindMemory(to: UInt8.self, capacity: width * height)
    else {
      return nil
    }
    
    for index in 0 ..< width * height {
      buffer[index] = data[index]
    }
    
    let image = context.makeImage().flatMap { UIImage(cgImage: $0) }
    
    return image
  }
  
}
#endif
