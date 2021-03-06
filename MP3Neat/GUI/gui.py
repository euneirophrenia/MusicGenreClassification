import wx
from os import path as p
import os
import subprocess
import datatools

import matplotlib
matplotlib.use('WXAgg')

import builtins
from Utility import statistics as statistics


datasets = ['./Datasets/MIDI/Binary/'+file for file in os.listdir('./Datasets/MIDI/Binary') if '_' not in file]

class SettingsDialog(wx.Dialog):
    def __init__(self, settings, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        self.settings = settings


        self.panel = wx.Panel(self)
        self.button_ok, self.button_cancel = self._configureButtons()

        self.comboboxes = []
        self.sizer = wx.GridSizer(cols = 2, rows = len(self.settings)+1, hgap=5, vgap=3)

        for i, key in enumerate(sorted(self.settings)):
            if all(type(x)==bool for x in self.settings[key][0]):
                combobox = wx.CheckBox(self.panel, id=wx.ID_ANY)
                combobox.SetValue(bool(self.settings[key][1]))
            else:
                combobox = wx.ComboBox(self.panel, id=wx.ID_ANY, value=str(self.settings[key][1]), choices = self.settings[key][0],
                                   style = wx.CB_READONLY, name=key)

            combobox.Enable(self.settings[key][2] if len(self.settings[key])>2 else True)
            self.comboboxes.append(combobox)

            self.sizer.Add(wx.StaticText(self, id=wx.ID_ANY, label=key, style=wx.ALIGN_LEFT), 0, wx.EXPAND| wx.ALL, border=5)
            self.sizer.Add(combobox, 0, wx.EXPAND| wx.ALL, border=5)


        self.sizer.Add(self.button_ok, 0, wx.EXPAND| wx.ALL, border=5)
        self.sizer.Add(self.button_cancel, 0, wx.EXPAND | wx.ALL, border=5)

        self.panel.SetSizerAndFit(self.sizer)
        self.Fit()
        self.CentreOnScreen()


    def _configureButtons(self):
        button_ok = wx.Button(self.panel, label="Ok")
        button_cancel = wx.Button(self.panel, label="Cancel")
        button_ok.Bind(wx.EVT_BUTTON, self.onOk)
        button_cancel.Bind(wx.EVT_BUTTON, self.onCancel)
        return button_ok, button_cancel


    def onCancel(self, _):
        self.EndModal(wx.ID_CANCEL)

    def onOk(self, _):
        for i, key in enumerate(sorted(self.settings)):
            self.settings[key] = (self.settings[key][0],self.comboboxes[i].GetValue())
        self.EndModal(wx.ID_OK)

    def GetSettings(self):
        return self.settings



class MainGUI(wx.Frame):
    ASKMIDIS = 39
    SETTINGS = 13
    PLOT = 12

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "MIDI Genre Classification")


        self.settings = {'Output dimension selection criterium':(['min','max'], 'min') , 'File type':(['MIDI'],'MIDI', False),
                         'Path to library' : (['./Utility/JSymbolic/jSymbolic2.jar'],'./Utility/JSymbolic/jSymbolic2.jar',False),
                         'Register path' :(['./saferegister.dat'],'./saferegister.dat',False),
                         'Audio playing command' : (['timidity'],'timidity',False)}

        panel = wx.Panel(self, wx.ID_ANY)
        self.index = 0

        self.nameToPath={}


        self.SetSize(905, 320)
        self.list_ctrl = wx.ListCtrl(panel, size=(900, 300),
                                     style=wx.LC_REPORT |
                                           wx.BORDER_SUNKEN |
                                           wx.LC_HRULES |
                                           wx.LC_VRULES
                                     )
        self.list_ctrl.InsertColumn(0, 'Title', width=425)
        self.list_ctrl.InsertColumn(1, 'Inferred genre', width=155)
        self.list_ctrl.InsertColumn(2, 'Confidence', width=225)

        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnPlayBack, self.list_ctrl)

        self.menubar = wx.MenuBar()
        self.mainmenu = wx.Menu()

        btn = wx.MenuItem(self.mainmenu, MainGUI.ASKMIDIS, "&Select MIDIs\tCTRL+O")
        setts = wx.MenuItem(self.mainmenu, MainGUI.SETTINGS, "&Settings\tCTRL+H")
        plots = wx.MenuItem(self.mainmenu, MainGUI.PLOT, "&Plot Features Rank\tCTRL+P")

        self.mainmenu.Append(btn)
        self.mainmenu.Append(setts)
        self.mainmenu.Append(plots)

        self.Bind(wx.EVT_MENU, self.askForMidis, id=MainGUI.ASKMIDIS)
        self.Bind(wx.EVT_MENU, self.Settings, id=MainGUI.SETTINGS)
        self.Bind(wx.EVT_MENU, self.Plots, id=MainGUI.PLOT)

        self.menubar.Append(self.mainmenu, '&Menu')
        self.SetMenuBar(self.menubar)

        self.Center()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.list_ctrl, 2, wx.ALL | wx.EXPAND, border=5)
        panel.SetSizer(sizer)

        self.CentreOnScreen()

    def add_line(self, path, inferredgenre):
        self.nameToPath[p.basename(path)]=path
        self.list_ctrl.InsertItem(self.index, p.basename(path), wx.LIST_FORMAT_CENTER)
        self.list_ctrl.SetItem(self.index, 1, str(inferredgenre[0]), wx.LIST_FORMAT_CENTER)
        self.list_ctrl.SetItem(self.index, 2, str(1-inferredgenre[1]), wx.LIST_FORMAT_CENTER)
        self.index += 1

    def askForMidis(self, _):
        openFileDialog = wx.FileDialog(self, "Open", "", "",
                                       "MIDI (*.midi)|*.mid",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)

        res = openFileDialog.ShowModal()
        if res == wx.ID_OK:
            files = openFileDialog.GetPaths()
            openFileDialog.Destroy()
            classifier = datatools.MIDIMiner(pathtolib = self.settings['Path to library'][1])
            result = classifier.classify(files, orderSelectionCriterium=getattr(builtins,self.settings['Output dimension selection criterium'][1]),
                                         register=self.settings['Register path'][1])[0]
            for key in result:
                self.add_line(key, result[key])

    def OnPlayBack(self, event):
        ind = event.GetIndex()
        name= self.list_ctrl.GetItem(ind, 0).GetText()
        path = self.nameToPath[name]
        command = [self.settings['Audio playing command'][1], path]

        dlg = wx.MessageDialog(self, 'Close this to stop', 'Playing '+name, style=wx.ICON_EXCLAMATION)

        processo= subprocess.Popen(command)

        dlg.ShowModal()

        processo.kill()

        dlg.Destroy()

    def Settings(self, _):
        box = SettingsDialog(self.settings, self, title='Settings', style=wx.RESIZE_BORDER | wx.CLOSE_BOX)
        box.ShowModal()
        self.settings = box.GetSettings()


    def Plots(self, _):
        dialog = SettingsDialog({'Dataset':(datasets, datasets[0]), 'Output dimension':(['1','2','3'],1),
                             'Algorithm':(['NEAT standard', 'Mutating Training Set NEAT'], 'NEAT Standard'),
                                 'Use histograms':([False],False, False)}, self, title='Plot Ranks',
                                style=wx.RESIZE_BORDER|wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        wantplot = dialog.ShowModal()

        if wantplot == wx.ID_OK:
            settings = dialog.GetSettings()
            didActuallyShow = statistics.plotRank(settings['Dataset'][1], settings['Algorithm'][1],
                                                  int(settings['Output dimension'][1]))

            if not didActuallyShow:
                wx.MessageBox(message='No run found for this configuration', parent=self, caption='Error occured')



if __name__ == '__main__':
    try:
        app = wx.App(False)
        frame = MainGUI()
        frame.Show()
        app.MainLoop()
    except Exception:
        pass


