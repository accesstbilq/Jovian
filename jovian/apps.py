from django.apps import AppConfig


class JovianConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'jovian'

    def ready(self):
        print("🔥 Jovian APP LOADED")
