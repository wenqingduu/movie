import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretest'))
from facelift_rgba_compat import rgb_fallback,sha

class FormatCompatibility(unittest.TestCase):
    def test_rgb_input_is_identical_object(self):
        image=Image.new('RGB',(4,4),(13,24,35))
        self.assertIs(rgb_fallback(image),image)

    def test_transparency_composites_to_original_white_background(self):
        image=Image.new('RGBA',(3,1));image.putdata([(10,20,30,0),(10,20,30,255),(10,20,30,128)])
        result=rgb_fallback(image)
        expected=Image.alpha_composite(Image.new('RGBA',image.size,(255,255,255,255)),image).convert('RGB')
        self.assertEqual(result.mode,'RGB');self.assertEqual(result.tobytes(),expected.tobytes())
        self.assertEqual(result.getpixel((0,0)),(255,255,255));self.assertEqual(result.getpixel((1,0)),(10,20,30))

    def test_provenance_binds_reference_and_input_pixels_and_resumes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reference=root/'reference.png';Image.new('RGB',(4,4),'red').save(reference)
            image=Image.new('RGBA',(4,4),(10,20,30,100));audit=root/'audit'
            with mock.patch.dict(os.environ,ENTITYBENCH_RGBA_SOURCE_REFERENCE=str(reference),ENTITYBENCH_RGBA_SOURCE_SHA256=sha(reference)):
                result=rgb_fallback(image,audit);rgb_fallback(image,audit)
                record=json.loads(next(audit.glob('*.json')).read_text())
                self.assertEqual(sha(record['after']),record['after_sha256'])
                self.assertEqual(Image.open(record['after']).tobytes(),result.tobytes())
                Image.new('RGB',(4,4),'blue').save(reference)
                with self.assertRaises(AssertionError):rgb_fallback(image,audit)

if __name__=='__main__':unittest.main()
