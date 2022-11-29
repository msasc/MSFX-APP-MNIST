/*
 * Copyright (c) 2022 Miquel Sas.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * 
 */
package com.msfx.app.mnist;

import java.io.File;
import java.util.List;

import com.msfx.lib.fx.Alert;
import com.msfx.lib.ml.function.Activation;
import com.msfx.lib.ml.graph.Cell;
import com.msfx.lib.ml.graph.Graph;
import com.msfx.lib.ml.graph.Network;
import com.msfx.lib.util.Files;
import com.msfx.lib.util.res.FileStringRes;
import com.msfx.lib.util.res.StringRes;

import javafx.application.Application;
import javafx.stage.Stage;

/**
 * MNIST trainer and tester.
 * 
 * @author Miquel Sas
 */
public class Trainer extends Application {
	
	public static void main(String[] args) { launch(args); }

	@Override
	public void start(Stage stage) throws Exception {

		/* Setup strings. */
		
		StringRes.setDefault(new FileStringRes("StringsLibrary.xml"));
		
		/* Files train and test. */
		
		File fileTrainImages = Files.findFileWithinClassPathEntries("train-images.idx3-ubyte.mnist_images");
		File fileTrainLabels = Files.findFileWithinClassPathEntries("train-labels.idx1-ubyte.mnist_labels");
		File fileTestImages = Files.findFileWithinClassPathEntries("t10k-images.idx3-ubyte.mnist_images");
		File fileTestLabels = Files.findFileWithinClassPathEntries("t10k-labels.idx1-ubyte.mnist_labels");
		
		boolean fileError = false;
		fileError |= fileTrainImages == null;
		fileError |= fileTrainLabels == null;
		fileError |= fileTestImages == null;
		fileError |= fileTestLabels == null;
		if (fileError) {
			Alert alert = new Alert(0.5, 0.2);
			alert.setup(Alert.Type.ERROR, Alert.Content.HTML);
			alert.setTitle("File error");
			alert.addHeaderText("Train and/or test files not found", "-fx-font-size: 20;");
			alert.show();
			return;
		}
		
		/* Read patterns. */
		
		List<MNIST> patternsTrain = MNIST.read(fileTrainLabels, fileTrainImages);
		List<MNIST> patternsTest = MNIST.read(fileTestLabels, fileTestImages);
		
		/* Build the network. */
		
		Network network = new Network();
		Cell cellIn = Graph.rnn(MNIST.ROWS * MNIST.COLS, 256, Activation.SIGMOID, false, true);
		Cell cellOut = Graph.rnn(256, 10, Activation.SIGMOID, false, true);
		Graph.connect(cellIn, cellOut);
		network.add(cellIn, cellOut);
		
		System.exit(0);
	}
}
