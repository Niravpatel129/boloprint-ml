const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const Human = require('@vladmandic/human');
const multer = require('multer');

const upload = multer({ dest: 'uploads/' });

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
  const tensor = human.tf.node.decodeImage(imageBuffer);
  const result = await human.detect(tensor);
  human.tf.dispose(tensor);

  // Create a mask based on detected body parts
  const mask = new Uint8Array(width * height).fill(0);
  result.body.forEach((body) => {
    body.keypoints.forEach((point) => {
      const x = Math.floor(point.x * width);
      const y = Math.floor(point.y * height);
      const idx = y * width + x;
      mask[idx] = 255;
    });
  });

  // Create a mask image
  const maskImage = sharp(Buffer.from(mask), {
    raw: { width, height, channels: 1 },
  });

  // Simulate dilation by applying a blur and threshold
  const dilatedMask = await maskImage.blur(10).threshold(128).raw().toBuffer();

  // Create a new sharp image from the dilated mask
  const finalMaskImage = sharp(dilatedMask, {
    raw: { width, height, channels: 1 },
  }).resize(width, height);

  // Composite the original image with the mask to remove the background
  await image
    .composite([{ input: await finalMaskImage.png().toBuffer(), blend: 'dest-in' }])
    .png()
    .toFile(outputImagePath);

  console.log(`Background removed. Output saved to ${outputImagePath}`);
}

module.exports = [
  upload.single('image'),
  async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image file uploaded' });
      }

      const inputImagePath = req.file.path;
      const outputFileName = `bg_removed_${Date.now()}.png`;
      const outputImagePath = path.join('images', outputFileName);

      // Ensure the images directory exists
      if (!fs.existsSync('images')) {
        fs.mkdirSync('images');
      }

      await removeBackground(inputImagePath, outputImagePath);

      // Clean up the uploaded file
      fs.unlinkSync(inputImagePath);

      res.status(200).json({
        message: 'Background removed successfully',
        outputPath: outputImagePath,
      });
    } catch (error) {
      console.error('Error removing background:', error);
      res.status(500).json({ error: 'An error occurred while removing the background' });
    }
  },
];
