import os
import pathlib
from json import load, dump
import tarfile
import io
import locale
import pandas as pd


class Params:
    def __init__(self):
        self.data = {}


class ParamGroup:
    UNKNOWN_MODE_ERROR = 0
    UNKNOWN_MODE_IGNORE = 1
    UNKNOWN_MODE_ADD = 2

    def __init__(self, key, name=""):
        self.key = key
        self.name = name

        self.data = {}

    def addParam(self, key:str, name:str, unit:str="", default=None, fmt=None, fmtFunc=None):
        param = {'name':name, 'val':default, 'unit':unit, 'default':default, 'fmt':fmt, 'fmtFunc':fmtFunc}

        self.data[key] = param

    def getValue(self, key:str):
        param = self.data[key]
        return param['val']

    def getFmtValue(self, key:str, withUnit=False):
        param = self.data[key]
        val = param['val']
        s = ""
        if param['fmtFunc'] != None:
            s = param['fmtFunc'](val)
        elif param['fmt']:
            # s = format(val, param['fmt'])
            s = locale.format_string(param['fmt'], val)
        else:
            if isinstance(val, float):
                s = locale.format_string("%.1f", val)
            else:
                s = str(val)

        unit = param['unit']
        if withUnit and len(unit):
            s += " [%s]" % unit

        return s

    def getName(self, key:str, withUnit=True):
        param = self.data[key]
        name = param['name']
        unit = param['unit']
        if withUnit and len(unit):
            name += " [%s]" % unit
        return name


    def setValue(self, key:str, _val, mode = UNKNOWN_MODE_ERROR):
        param = self.data.get(key, None)
        if param is None:
            if mode == self.UNKNOWN_MODE_IGNORE:
                return
            elif mode == self.UNKNOWN_MODE_ADD:
                self.addParam(key, "", "")
                param = self.data[key]
            else:
                raise KeyError("key %s not included in params" % (key))

        param_val = param['val']
        if isinstance(param_val, float):
            if isinstance(_val, str):
                param_val = locale.atof(_val)
            else:
                param_val = float(_val)
        elif isinstance(param_val, bool):
            param_val = bool(_val)
        elif isinstance(param_val, int):
            param_val = int(_val)
        else:
            param_val = _val

        # print("setValue(%s = (%s) -> %s" % (key, str(_val), str(param_val)))
        param['val'] = param_val

    def getJSON(self):
        json_data = {}

        params = {}
        for key, param in self.data.items():
            params[key] = param['val']

        json_data[self.key] = params

        return json_data

    def saveJSON(self, fname):
        # Create path when not exists
        path = pathlib.PureWindowsPath(fname).parents[0]
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Write JSON file
        json_data = self.getJSON()
        with open(fname, 'w') as f:
            dump(json_data, f)

    def writeJSONStream(self, stream):
        json_data = self.getJSON()
        dump(json_data, stream)

    def getFName(self, prefix):
        return "%s.json" % (prefix)

    def writeToTar(self, tar, fname_prefix):
        # Stream JSON structure to String and then convert to Binary stream
        stream_str = io.StringIO()
        self.writeJSONStream(stream_str)
        stream = io.BytesIO(stream_str.getvalue().encode())
        stream.seek(0)

        # Add as file to tar
        info = tarfile.TarInfo(self.getFName(fname_prefix))
        info.size = len(stream.getvalue())
        tar.addfile(info, stream)

    def readFromTar(self, tar, tar_content, fname_prefix, add_new = False):
        fname = self.getFName(fname_prefix)
        if fname not in tar_content:
            return False
        f = tar.extractfile(fname)
        json_data = load(f)
        mode = self.UNKNOWN_MODE_ERROR
        if add_new:
            mode = self.UNKNOWN_MODE_ADD
        self.setJSON(json_data, mode)
        return True

    def loadJSON(self, fname):
        if not os.path.isfile(fname):
            return
        json_data = {}
        with open(fname, 'r') as f:
            json_data = load(f)
            self.setJSON(json_data)

    def setJSON(self, json_data, mode = UNKNOWN_MODE_ERROR):
        params = json_data.get(self.key, {})
        for key, value in params.items():
            self.setValue(key, value, mode)

    def getTable(self):
        table = []
        for key, param in self.data.items():
            table.append([param['name'], self.getFmtValue(key, withUnit=True)])

        return table

    def getPD(self):
        data = {'Parameter': [],
                'Value': [],
                'Unit': []}

        for key, param in self.data.items():
            data['Parameter'].append(param['name'])
            data['Value'].append(param['val'])
            data['Unit'].append(param['unit'])

        return pd.DataFrame(data)
