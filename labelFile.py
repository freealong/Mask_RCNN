#
# Copyright (C) 2011 Michael Pitidis, Hussein Abdulwahid.
#
# This file is part of Labelme.
#
# Labelme is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Labelme is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Labelme.  If not, see <http://www.gnu.org/licenses/>.
#

from base64 import b64encode, b64decode
import json
import os.path

class LabelFileError(Exception):
    pass

class LabelFile(object):
    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        if filename is not None:
            self.load(filename)

    def load(self, filename, data_path=None):
        try:
            with open(filename) as f:
                data = json.load(f)
                width = data['width']
                height = data['height']
                imagePath = data['imagePath']
                lineColor = data['lineColor']
                fillColor = data['fillColor']
                shapes = ((s['label'], s['points'], s['line_color'], s['fill_color'])\
                        for s in data['shapes'])
                # Only replace data after everything is loaded.
                self.shapes = shapes
                if data_path is None:
                    self.imagePath = imagePath
                else:
                    self.imagePath = os.path.join(data_path, os.path.split(imagePath)[-1])
                self.lineColor = lineColor
                self.fillColor = fillColor
                self.width = width
                self.height = height
        except Exception as e:
            raise LabelFileError(e)

    def save(self, filename, shapes, imagePath,
             width, height,
            lineColor=None, fillColor=None):
        try:
            with open(filename, 'wb') as f:
                json.dump(dict(
                    width=width, height=height,
                    shapes=shapes,
                    lineColor=lineColor, fillColor=fillColor,
                    imagePath=imagePath),
                    f, ensure_ascii=True, indent=2)
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def isLabelFile(filename):
        return os.path.splitext(filename)[1].lower() == LabelFile.suffix
