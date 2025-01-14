from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from .image_search import Search_Setup
from .image_verify import Search_Vertify
from django.views.decorators.csrf import csrf_protect
import os

def index(request):
    return render(request, 'image_search/index.html')

@csrf_protect
def search_images(request):
    if request.method == 'POST':
        # Handle file uploads
        if 'images[]' in request.FILES:
            fs = FileSystemStorage()
            uploaded_files = request.FILES.getlist('images[]')
            image_paths = []
            
            for file in uploaded_files:
                filename = fs.save(f'uploads/{file.name}', file)
                image_paths.append(fs.path(filename))
            
            # Set up search engine
            batch_size = 16
            ss = Search_Setup(
                image_list=image_paths,
                model_name='openai/clip-vit-base-patch16'
            )
            ss.run_index(batch_size=batch_size)
            
            # Store the search setup in session
            request.session['image_paths'] = image_paths
            
            return JsonResponse({"status": 'success'})
            
        # Handle search query
        elif 'query' in request.POST:
            query = request.POST.get('query')
            number = int(request.POST.get('number', 1))
            image_paths = request.session.get('image_paths', [])
            
            if not image_paths:
                return JsonResponse({"error": 'No images uploaded'})
            
            # search for images
            ss = Search_Setup(
                image_list=image_paths,
                model_name='openai/clip-vit-base-patch16'
            )
            # ss.run_index()
            results = ss.get_images_by_text(text=query, number_of_images=number)

            # identify matched entities in query
            sv = Search_Vertify()
            for r in results:
                matched = sv.match(r["image"], query)
                matched = ", ".join(matched) if len(matched) > 0 else "NA"
                r["matched"] = matched
            return JsonResponse({'results': results})
            
    return JsonResponse({"error": 'Invalid request'}) 