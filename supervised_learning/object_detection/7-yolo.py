    def predict(self, folder_path):
        """Predict boxes for all images in a folder"""
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []

        for i, image in enumerate(images):
            image_outputs = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs, image_shapes[i]
            )

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )

            pred_boxes, pred_classes, pred_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores
            )

            predictions.append((pred_boxes, pred_classes, pred_scores))

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(image, pred_boxes, pred_classes,
                            pred_scores, file_name)

        return predictions, image_paths
