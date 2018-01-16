#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Copyright Université de Nantes

Created on Tue Jun 20 2017
@author : Harold Mouchère

@contact : Harold Mouchère

Piff class to load, save and access Piff files

"""

import json

from io import StringIO

class Piff:

    def __init__(self):
        self.meta = {}
        self.location = []
        self.data = []


    def new_data_process(self, data_id, location_id, content, parent_id_list, data_type):
        data = {}
        data['id'] = data_id
        data['location_id'] = location_id
        data['type'] = data_type
        data['value'] = content

        if parent_id_list is not None :
            data['parent_id_list'] = parent_id_list
        return data

    def add_elt_data(self, data_id, location_id, content, parent_id_list, data_type):
        self.data.append(self.new_data_process(data_id, location_id, content, parent_id_list, data_type))

    def new_location_process(self,location_id, data_type, polygon):
        location = {}
        location['id'] = location_id
        location['type'] = data_type
        location['polygon'] = polygon
        return location

    def add_elt_location(self, location_id, location_type, polygon):
        self.location.append(self.new_location_process(location_id, location_type, polygon))

    def load(self, filename):
        file = open(filename, 'r',encoding='utf-8')
        jsdict = json.load(file)
        file.close()
        self.meta = jsdict["meta"]
        self.location = jsdict["location"]
        self.data = jsdict["data"]

    def __str__(self):
        io = StringIO()
        json.dump(self, io, cls=PiffJSONEncoder, indent=1)
        return io.getvalue()

    def save(self, filename):
        file = open(filename, 'w')
        json.dump(self, file, cls=PiffJSONEncoder, indent=1, ensure_ascii=False)
        file.close()

    def get_location(self,location_id):
        for elt_location in self.location :
            if elt_location['id'] == location_id :
                return elt_location

class PiffJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Piff):
            return {"meta": obj.meta, "location" : obj.location, "data" : obj.data}
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
