const fs = require('fs');
const sharp = require('sharp');
const Human = require('@vladmandic/human');

async function removeBackground(inputImagePath, outputImagePath) {
  // Load and configure Human
  const human = new Human.Human({
    backend: 'tensorflow',
    modelBasePath: 'https://vladmandic.github.io/human/models',
    face: { enabled: false },
    body: { enabled: true },
    hand: { enabled: false },
    object: { enabled: false },
  });

  // Load the input image
  const imageBuffer = fs.readFileSync(inputImagePath);
  const image = sharp(imageBuffer);
  const { width, height } = await image.metadata();

  // Detect the subject (body)
  const tensor = human.tf.node.decodeImage(imageBuffer, 3);
  const result = await human.detect(tensor);
  tensor.dispose();

  // Create a mask based on detected body parts
  const mask = new Uint8Array(width * height * 4);
  result.body.forEach((body) => {
    body.keypoints.forEach((point) => {
      const x = Math.floor(point.position[0]);
      const y = Math.floor(point.position[1]);
      const idx = (y * width + x) * 4;
      mask[idx] = 255; // R
      mask[idx + 1] = 255; // G
      mask[idx + 2] = 255; // B
      mask[idx + 3] = 255; // A
    });
  });

  // Create a mask image
  const maskImage = sharp(Buffer.from(mask), {
    raw: { width, height, channels: 4 },
  });

  // Composite the original image with the mask to remove the background
  await image
    .joinChannel(await maskImage.toBuffer(), { raw: { width, height, channels: 4 } })
    .removeAlpha()
    .toFile(outputImagePath);

  console.log(`Background removed. Output saved to ${outputImagePath}`);
}

const inputImagePath = 'path/to/your/input/image.jpg';
const outputImagePath = 'path/to/your/output/image.png';
removeBackground(inputImagePath, outputImagePath).catch(console.error);
