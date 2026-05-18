const API_BASE_URL = 'http://localhost:8000';

const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const autocompleteDropdown = document.getElementById('autocompleteDropdown');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const recommendationsGrid = document.getElementById('recommendationsGrid');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

let allBooks = [];

// Fetch book list for autocomplete on load
async function fetchBooks() {
    try {
        const response = await fetch(`${API_BASE_URL}/books`);
        const data = await response.json();
        allBooks = data.books || [];
    } catch (err) {
        console.error("Failed to fetch books for autocomplete", err);
    }
}

fetchBooks();

// Handle Autocomplete
searchInput.addEventListener('input', (e) => {
    const value = e.target.value.toLowerCase();
    autocompleteDropdown.innerHTML = '';
    
    if (!value) {
        autocompleteDropdown.classList.add('hidden');
        return;
    }

    const matches = allBooks.filter(book => book.toLowerCase().includes(value)).slice(0, 10);
    
    if (matches.length > 0) {
        autocompleteDropdown.classList.remove('hidden');
        matches.forEach(match => {
            const div = document.createElement('div');
            div.className = 'autocomplete-item';
            div.textContent = match;
            div.addEventListener('click', () => {
                searchInput.value = match;
                autocompleteDropdown.classList.add('hidden');
                fetchRecommendations(match);
            });
            autocompleteDropdown.appendChild(div);
        });
    } else {
        autocompleteDropdown.classList.add('hidden');
    }
});

document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !autocompleteDropdown.contains(e.target)) {
        autocompleteDropdown.classList.add('hidden');
    }
});

// Search execution
searchButton.addEventListener('click', () => {
    if (searchInput.value.trim()) {
        fetchRecommendations(searchInput.value.trim());
    }
});

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && searchInput.value.trim()) {
        autocompleteDropdown.classList.add('hidden');
        fetchRecommendations(searchInput.value.trim());
    }
});

async function fetchRecommendations(title) {
    loadingSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    recommendationsGrid.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE_URL}/recommend?title=${encodeURIComponent(title)}`);
        const data = await response.json();

        loadingSection.classList.add('hidden');

        if (data.error) {
            showError("We couldn't find recommendations for that book. Please try another one!");
            return;
        }

        renderRecommendations(data.recommendations);
    } catch (err) {
        loadingSection.classList.add('hidden');
        showError("Something went wrong. Please ensure the backend server is running.");
        console.error(err);
    }
}

function renderRecommendations(books) {
    if (!books || books.length === 0) {
        showError("No recommendations found.");
        return;
    }

    resultsSection.classList.remove('hidden');

    books.forEach((book, index) => {
        const card = document.createElement('div');
        card.className = 'book-card';
        card.style.animationDelay = `${index * 0.1}s`;

        // Format author string if it's an array representation
        let authorStr = book.author;
        try {
            if (typeof authorStr === 'string' && authorStr.startsWith('[')) {
                authorStr = JSON.parse(authorStr.replace(/'/g, '"')).join(', ');
            }
        } catch(e){}

        // Parse genres carefully
        let genreArray = [];
        try {
            if (book.genres.startsWith('[')) {
                 const parsed = JSON.parse(book.genres.replace(/'/g, '"'));
                 if(Array.isArray(parsed)) genreArray = parsed.slice(0, 5); // take up to 5 genres
            } else {
                 genreArray = book.genres.split(',').map(g => g.trim()).slice(0, 5);
            }
        } catch(e) {
             genreArray = book.genres.split(',').map(g => g.trim()).slice(0, 5);
        }

        const genresHtml = genreArray.map(g => `<span class="genre-tag">${g}</span>`).join('');

        card.innerHTML = `
            <div class="book-title">${book.title}</div>
            <div class="book-author">by ${authorStr}</div>
            <div class="book-rating">★ ${parseFloat(book.rating).toFixed(2)}</div>
            <div class="book-genres">${genresHtml}</div>
        `;
        
        recommendationsGrid.appendChild(card);
    });
}

function showError(msg) {
    errorSection.classList.remove('hidden');
    errorMessage.textContent = msg;
}
