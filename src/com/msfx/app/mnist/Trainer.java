/*
 * Copyright (c) 2022 Miquel Sas.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
import java.util.ArrayList;
import java.util.List;

import com.msfx.lib.fx.Alert;
import com.msfx.lib.fx.Buttons;
import com.msfx.lib.fx.Frame;
import com.msfx.lib.fx.PaneProgress;
import com.msfx.lib.ml.data.ListPatternSource;
import com.msfx.lib.ml.function.Activation;
import com.msfx.lib.ml.graph.Cell;
import com.msfx.lib.ml.graph.Graph;
import com.msfx.lib.ml.graph.Network;
import com.msfx.lib.ml.training.SLTrainer;
import com.msfx.lib.util.Files;
import com.msfx.lib.util.res.FileStringRes;
import com.msfx.lib.util.res.StringRes;

import javafx.application.Application;
import javafx.scene.control.Button;
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

		ListPatternSource sourceTrain = MNIST.read(fileTrainLabels, fileTrainImages);
		ListPatternSource sourceTest = MNIST.read(fileTestLabels, fileTestImages);

		/* Build the network. */

		Network network = new Network();
		int[] sizes = new int[] {
				MNIST.ROWS * MNIST.COLS, 256, 10
		};
		List<Cell> cells = new ArrayList<>();
		for (int i = 1; i < sizes.length; i++) {
			int sizeIn = sizes[i-1];
			int sizeOut = sizes[i];
			Cell cell = Graph.rnn(sizeIn, sizeOut, Activation.SIGMOID, false, true);
			cells.add(cell);
		}
		for (int i = 1; i < cells.size(); i++) {
			Cell cellIn = cells.get(i-1);
			Cell cellOut = cells.get(i);
			Graph.connect(cellIn, cellOut);
		}
		network.add(cells);
		network.setParallelProcessing(true);

		/* Build the trainer. */

		SLTrainer trainer = new SLTrainer();
		trainer.setNetwork(network);
		trainer.setSourceTrain(sourceTrain);
		trainer.setSourceTest(sourceTest);
		trainer.setNetwork(network);
		trainer.setEpochs(200);

		StringBuilder trainerTitle = new StringBuilder();
		for (int i = 0; i < cells.size(); i++) {
			if (i > 0) trainerTitle.append("-");
			trainerTitle.append(cells.get(i).getName());
		}
		trainer.setTitle(trainerTitle.toString());

		/* Build the frame. */

		Frame frame = new Frame(stage);

		Button close = Buttons.close(true, true, true);
		close.setOnAction(e -> {
			if (trainer.isRunning()) {
				Alert alert = new Alert(0.5, 0.2);
				alert.setup(Alert.Type.WARNING, Alert.Content.TEXT);
				alert.setTitle("Warning!");
				alert.addHeaderText("Task is still running, can not close the dialog",
					"-fx-font-size:14;-fx-font-weight:bold;");
				alert.show();
				e.consume();
			}
			if (trainer.hasTerminated()) {
				network.terminate();
			}
		});

		PaneProgress pane = new PaneProgress(trainer);
		pane.setRemoveAction(e -> frame.getPaneCombo().setCenter(null));
		trainer.setConsole(pane.getConsole());

		frame.getPaneCombo().getButtonBar().getButtons().add(close);
		frame.getPaneCombo().setCenter(pane.getRoot());

		frame.getStage().setTitle("Test progress pane");
		frame.sizeAndCenter(0.8, 0.8);
		frame.show();
	}
}
