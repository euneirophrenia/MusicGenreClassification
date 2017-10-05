import wx
import demo
from os import path as p
import subprocess
import datatools


class MainGUI(wx.Frame):
    ASKMIDIS = 39

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "MIDI Genre Classification")

        self.classifier = datatools.MIDIExtractor()

        panel = wx.Panel(self, wx.ID_ANY)
        self.index = 0

        self.nameToPath={}


        self.SetSize(900, 300)
        self.list_ctrl = wx.ListCtrl(panel, size=(900, 300),
                                     style=wx.LC_REPORT
                                           | wx.BORDER_SUNKEN
                                     )
        self.list_ctrl.InsertColumn(0, 'Title', width=225)
        self.list_ctrl.InsertColumn(1, 'Inferred genre', width=155)
        self.list_ctrl.InsertColumn(2, 'Confidence', width=225)

        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnPlayBack, self.list_ctrl)

        self.menubar = wx.MenuBar()
        self.mainmenu = wx.Menu()

        btn = wx.MenuItem(self.mainmenu, MainGUI.ASKMIDIS, "&Select MIDIs\tCTRL+O")

        self.mainmenu.Append(btn)

        self.Bind(wx.EVT_MENU, self.askForMidis, id=MainGUI.ASKMIDIS)

        self.menubar.Append(self.mainmenu, '&The only menu I will EVER add')
        self.SetMenuBar(self.menubar)

        self.Center()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.list_ctrl, 2, wx.ALL | wx.EXPAND, 5)
        panel.SetSizer(sizer)

    def add_line(self, path, inferredgenre):
        self.nameToPath[p.basename(path)]=path
        self.list_ctrl.InsertItem(self.index, p.basename(path))
        self.list_ctrl.SetItem(self.index, 1, str(inferredgenre[0]))
        self.list_ctrl.SetItem(self.index, 2, str(1-inferredgenre[1]))
        self.index += 1

    def askForMidis(self, _):
        openFileDialog = wx.FileDialog(self, "Open", "", "",
                                       "MIDI (*.midi)|*.mid",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)

        openFileDialog.ShowModal()

        files = openFileDialog.GetPaths()
        openFileDialog.Destroy()
        result = self.classifier.classify(files, orderSelectionCriterium=max, register='./saferegister.dat')[0]

        for key in result:
            self.add_line(key, result[key])

    def OnPlayBack(self, event):
        ind = event.GetIndex()
        name= self.list_ctrl.GetItem(ind, 0).GetText()
        path = self.nameToPath[name]
        command = ['timidity', path]

        dlg = wx.MessageDialog(self, 'Clicca per fermare', 'Playing '+name)

        processo= subprocess.Popen(command)

        dlg.ShowModal()

        processo.kill()

        dlg.Destroy()




try:
    app = wx.App(False)
    frame = MainGUI()
    frame.Show()
    app.MainLoop()
except Exception:
    pass


