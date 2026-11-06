"""Create the portable launcher's native startup bitmap from the app's UI colors."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
font_dir = Path('C:/Windows/Fonts')

def font(size, bold=False):
    path = font_dir / ('segoeuib.ttf' if bold else 'segoeui.ttf')
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.truetype('DejaVuSans.ttf', size)

image = Image.new('RGB', (480, 250), '#e8e9e8')
draw = ImageDraw.Draw(image)
draw.rectangle((0, 0, 479, 249), outline='#c2cbc0', width=1)
draw.rounded_rectangle((207, 27, 273, 93), radius=16, fill='#c9bedf')
draw.text((221, 30), 'S', font=font(43, True), fill='#544569')
draw.ellipse((252, 71, 258, 77), fill='#544569')
draw.text((240, 115), 'Starting Spyder Cal', font=font(24, True), fill='#303b34', anchor='mt')
draw.text((240, 157), 'Preparing the application...', font=font(15), fill='#69756c', anchor='mt')
for x, color in [(225, '#c9bedf'), (240, '#bedaca'), (255, '#c6d9e1')]:
    draw.ellipse((x-3, 201, x+3, 207), fill=color)
(root / 'assets').mkdir(exist_ok=True)
image.save(root / 'assets/startup.bmp')
