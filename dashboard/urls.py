from django.urls import path
from .views import (
    DASHBOARD,
    PROFILE,
    HISTORY,
    Feedback_Done,
    CHANGEPASSWORD,
    detailed,
    FEEDBACK,
    DeleteGroup,
    UPDATTEXT,
    UpdateGroup,
)

urlpatterns = [
    path("dashboard", DASHBOARD.as_view(), name="dashboard"),
    path("profile", PROFILE.as_view(), name="profile"),
    path("history", HISTORY.as_view(), name="history"),
    path("feedback", FEEDBACK.as_view(), name="feedback"),
    path("feedback_done", Feedback_Done.as_view(), name="feedback_done"),
    path("changepassword", CHANGEPASSWORD.as_view(), name="changepassword"),
    path("details/<str:group>/", detailed.as_view(), name="details"),
    path('history/<str:group>/update/', UpdateGroup.as_view(), name='update_group'),
    
    path('history/<str:group>/delete/', DeleteGroup.as_view(), name='delete_group'),
    path("updatetext<int:pk>", UPDATTEXT.as_view(), name="updatetext"),
]
