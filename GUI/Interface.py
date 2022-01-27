from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

kv = Builder.load_file('interface.kv')


class LegWindow(Screen):
    pass

class Home(Screen):
    pass


class InterfaceApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(Home(name="Home"))
        sm.add_widget(LegWindow(name="LegScreen"))
        return sm

if __name__ == '__main__':
    InterfaceApp().run()