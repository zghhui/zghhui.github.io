
function copyBibtex() {
    const bibtexCode = document.querySelector('.bibtex-container pre code');
    const text = bibtexCode.innerText;
    navigator.clipboard.writeText(text).then(() => {
        const copyBtn = document.querySelector('.copy-btn');
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}