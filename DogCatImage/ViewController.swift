//
//  ViewController.swift
//  DogCatImage
//
//  Created by nakamori on 2022/01/18.
//

import UIKit
import Vision
import CoreML

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, UIGestureRecognizerDelegate {
 
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var label: UILabel!
    
    var model = try! VNCoreMLModel(for: MobileNetV2().model)
    
    //ビュー表示時に呼ばれる
    override func viewDidAppear(_ animated: Bool) {
        if self.imageView.image == nil {
            showActionSheet()
        }
    }
    
    //画面タップ時に呼ばれる
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        showActionSheet()
    }
    
    //アクションシートの表示
    func showActionSheet() {
        let actionSheet = UIAlertController(title: nil, message: nil, preferredStyle: .actionSheet)
        actionSheet.addAction(UIAlertAction(title: "カメラ", style: .default) { action in
            self.openPicker(sourceType: .camera)
        })
        actionSheet.addAction(UIAlertAction(title: "フォトライブラリ", style: .default) { action in
            self.openPicker(sourceType: .photoLibrary)
        })
        actionSheet.addAction(UIAlertAction(title: "キャンセル", style: .cancel))
        self.present(actionSheet, animated: true, completion: nil)
    }
    
    //アラートの表示
    func showAlert(_ text: String) {
        let alert = UIAlertController(title: text, message: nil, preferredStyle: UIAlertController.Style.alert)
        alert.addAction(UIAlertAction(title: "ok", style: UIAlertAction.Style.default, handler: nil))
        self.present(alert, animated: true, completion: nil)
    }
    
    //イメージピッカーのオープン
    func openPicker(sourceType: UIImagePickerController.SourceType) {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = self
        self.present(picker, animated: true, completion: nil)
        
    }
    //イメージピッカーのイメージ取得時に呼ばれる
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        var image = info[UIImagePickerController.InfoKey.originalImage] as! UIImage
        
        //画像向きの補正
        let size = image.size
        UIGraphicsBeginImageContext(size)
        image.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        //イメージの指定
        self.imageView.image = image
        //クローズ
        picker.presentingViewController!.dismiss(animated: true, completion: nil)
        //予測
        predict(image)
    }
    
    //イメージピッカーのキャンセル時に呼ばれる
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.presentingViewController?.dismiss(animated: true, completion: nil)
    }
    
    //予測
    func predict(_ image: UIImage) {
        DispatchQueue.global(qos: .default).async {
            //リクエストの生成
            let request = VNCoreMLRequest(model: self.model) { request, error in
                if error != nil {
                    self.showAlert(error!.localizedDescription)
                    return
                }
                
                //検出結果の取得
                let observations = request.results as! [VNClassificationObservation]
                var text: String = "\n"
                for i in 0..<min(3, observations.count) {
                    let probabillity = Int(observations[i].confidence*100) //信頼度
                    let label = observations[i].identifier //ラベル
                    text += "\(label) : \(probabillity)%\n"
                }
                
                //UIの更新
                DispatchQueue.main.async {
                    self.label.text = text
                }
            }
            
            //入力画像のリサイズ指定
            request.imageCropAndScaleOption = .centerCrop
            
            //UIImageをCIImageに変換
            let ciImage = CIImage(image: image)!
            
            //画像の向きの取得
            let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue))!
            
            //ハンドラの生成と実行
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            guard (try? handler.perform([request])) != nil else {
                return
            }
            
        }
    }
    
}

