from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import chat_view, portfolio_view

urlpatterns = [
    # page
    path("", chat_view.index, name="home"),
    path("start-chat", chat_view.chatbot_view, name="chat"),
    path("api/chat", chat_view.chat_asistance, name="chat-stream"),

    path('api/vectors/', portfolio_view.VectorStoreAPIView.as_view(), name='vectors-list'),
    path('api/search/', portfolio_view.VectorSearchAPIView.as_view(), name='vector-search'),
    path('api/vectors/json/', portfolio_view.vector_data_json, name='vectors-json'),

]

# if settings.DEBUG:
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)