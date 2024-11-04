from django.shortcuts import render
from django.views.generic import (
    TemplateView,
    DetailView,
    ListView,
    UpdateView,
    DeleteView,
    CreateView,
)
from .models import RECORD, Feedback
from django.shortcuts import get_object_or_404, redirect
from django.views import View

from django.urls import reverse_lazy
from django.contrib.auth.views import PasswordChangeView
from django.contrib.auth.mixins import LoginRequiredMixin
from django import forms

# Create your views here.


class DASHBOARD(LoginRequiredMixin, TemplateView):
    template_name = "dashboard/userInput.html"


class PROFILE(LoginRequiredMixin, TemplateView):
    template_name = "dashboard/profile.html"


class FEEDBACK(LoginRequiredMixin, CreateView):
    model = Feedback
    template_name = "dashboard/feedback.html"
    fields = ["feedback"]
    success_url = "feedback_done"

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)


class Feedback_Done(LoginRequiredMixin, TemplateView):
    template_name = "dashboard/feedback_done.html"


class HISTORY(LoginRequiredMixin, ListView):
    template_name = "dashboard/history.html"
    model = RECORD
    context_object_name = "groups"

    def get_queryset(self):
        return RECORD.objects.values('group').distinct()


class detailed(LoginRequiredMixin, ListView):
    template_name = "dashboard/detailView.html"
    model = RECORD
    context_object_name = "records"

    def get_queryset(self):
        group = self.kwargs['group']
        return RECORD.objects.filter(group=group)

class DeleteGroup(LoginRequiredMixin, View):
    def post(self, request, group):
        RECORD.objects.filter(group=group).delete()
        return redirect('history')  # Replace 'history' with your history view name


class UpdateGroupForm(forms.Form):
    new_group = forms.CharField(max_length=100, required=True, label='New Group Name')

class UpdateGroup(LoginRequiredMixin, View):
    template_name = 'dashboard/update.html'

    def get(self, request, group):
        form = UpdateGroupForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request, group):
        form = UpdateGroupForm(request.POST)
        if form.is_valid():
            new_group = form.cleaned_data['new_group']
            RECORD.objects.filter(group=group).update(group=new_group)
            return redirect('history')  # Replace 'history' with your history view name
        return render(request, self.template_name, {'form': form})

class CHANGEPASSWORD(LoginRequiredMixin, PasswordChangeView):
    template_name = "dashboard/changepassword.html"
    success_url = reverse_lazy("dashboard")


class UPDATTEXT(LoginRequiredMixin, UpdateView):
    model = RECORD
    template_name = "dashboard/update.html"
    fields = ["title"]
    success_url = reverse_lazy("history")


class DELETETEXT(DeleteView):
    login_url = reverse_lazy("login")
    model = RECORD
    template_name = "dashboard/delete.html"
    success_url = reverse_lazy("history")
    context_object_name = "deletetext"
