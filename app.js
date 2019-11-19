const express = require('express')
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const promisify = require('util').promisify;

const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);
const app = express()
const port = 3000

app.get('/', async (req, res) => {
  const labels = (await readFile('labels.txt', 'utf8')).split('\n');


  // const model =
  // await tf.node.loadSavedModel('./2', ['serve'], 'serving_default');
  const modelUrl =
      'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
  const model = await tf.loadGraphModel(modelUrl);
  // const model = await tf.loadGraphModel('file://web_model/model.json');


  const image = await readFile('croppedDog.jpg');
  const buf = Buffer.from(image);
  const uint8array = new Uint8Array(buf);
  const imageTensor = await tf.node.decodeImage(uint8array);

  // const input = tf.image.cropAndResize(
  //     imageTensor.toFloat().expandDims(0), tf.tensor2d([0, 0, 1, 1], [1, 4]),
  //     tf.tensor1d([0], 'int32'), [224, 224], 'bilinear');

  // console.log(input.shape);
  // const encodedImage =
  //     await tf.node.encodeJpeg(input.reshape([224, 224, 3]), 'rgb');
  // await writeFile('croppedDog.jpg', encodedImage);

  // const input = tf.tensor([1, 2, 3]);
  const input = imageTensor.toFloat().expandDims(0);
  let start = tf.util.now();
  const outputTensor = model.predict(input);
  // const outputTensor = await model.executeAsync(input);
  let end = tf.util.now();
  let time = end - start;
  console.log('time 1: ', time);

  start = tf.util.now();
  model.predict(input);
  // await model.executeAsync(input);
  end = tf.util.now();
  time = end - start;
  console.log('time 2: ', time);

  start = tf.util.now();
  model.predict(input);
  // await model.executeAsync(input);
  end = tf.util.now();
  time = end - start;
  console.log('time 3: ', time);

  start = tf.util.now();
  model.predict(input);
  // await model.executeAsync(input);
  end = tf.util.now();
  time = end - start;
  console.log('time 4: ', time);

  start = tf.util.now();
  model.predict(input);
  // await model.executeAsync(input);
  end = tf.util.now();
  time = end - start;
  console.log('time 5: ', time);

  const output = await outputTensor.data()

  // const output =
  //     await (await model.executeAsync(imageTensor.toFloat().expandDims(0)))
  //         .data();
  // console.log(output);

  let probability = 0;
  let label = '';
  let index;
  for (let i = 0; i < 1000; i++) {
    if (output[i] > probability) {
      probability = output[i];
      label = labels[i];
      index = i;
    }
  }
  console.log(probability, label, index);
  model.dispose();
  res.send('Hello World!' + label);
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
