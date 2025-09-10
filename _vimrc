call plug#begin('~/.vim/plugged')

Plug 'fatih/vim-go', { 'do': ':GoUpdateBinaries' }
Plug 'nsf/gocode'

Plug 'SirVer/ultisnips'
Plug 'honza/vim-snippets'

Plug 'neoclide/coc.nvim', {'branch': 'release'}

Plug 'ctrlpvim/ctrlp.vim'
Plug 'vim-scripts/BufOnly.vim'

Plug 'junegunn/fzf.vim'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }

Plug 'preservim/nerdtree' 
Plug 'airblade/vim-gitgutter'

Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'

Plug 'ryanoasis/vim-devicons'

Plug 'uarun/vim-protobuf'

"Plug 'psf/black'
Plug 'shaoran/vim-ruff'

Plug 'andrewstuart/vim-kubernetes'
Plug 'towolf/vim-helm'

Plug 'tpope/vim-surround'
Plug 'tpope/vim-fireplace'

"Colors
"Plug 'fatih/molokai'
Plug 'morhetz/gruvbox'
Plug 'nanotech/jellybeans.vim'

Plug 'rust-lang/rust.vim'

call plug#end()

"
" ############ SETTINGS ###################
"set t_Co=256

" This must be first, because other otpions as a side effects.
set nocompatible      " Don't be compatible with vi

" Seriously, guys. It's not like :W is bound to anything anyway.
command! W :w

" Disable intro screen
set shm=atI

" Improve redrawing for newer computers
set ttyfast

" Disable backup
set nobk nowb noswf

"Only undo up to 150
set undolevels=150

" Allow backspacing over anything
set bs=2

" Allow backspacing over everything in insert mode
set backspace=indent,eol,start

" Store lots os :cmdline history
set history=1000

" Show incomplete cmds down the bottom
set showcmd

" Show current mode down the bottom
set showmode

" Disable visual bell
set vb

" Show breakline
set showbreak=...

" Vertical/horizontal scroll off setting
set scrolloff=3
set sidescrolloff=7
set sidescroll=1

" Wild Mode
" Make cmdline tab completation similar to bash
set wildmode=full
" Enable ctrl-n and ctrl-p to scroll thru matches
set wildmenu

" Ignore these files when completation
set wildignore+=*.o,*.obj,.git,*.pyc

"""" Messages, Info, Status
set ls=2                    " allways show status line
set vb t_vb=                " Disable all bells.  I hate ringing/flashing.
set confirm                 " Y-N-C prompt if closing with unsaved changes
set showcmd                 " Show incomplete normal mode commands as I type.
set report=0                " : commands always print changed line count
set shortmess+=a            " Use [+]/[RO]/[w] for modified/readonly/written
set ruler                   " Show some info, even without statuslines.
set laststatus=2            " Always show statusline, even if only 1 window.

" displays tabs with :set list & displays when a line runs off-screen
"set listchars=tab:>-,eol:$,trail:-,precedes:<,extends:>
"set list
set hidden

""" Searching and Patterns
set ignorecase              " Default to using case insensitive searches,
set smartcase               " unless uppercase letters are used in the regex.
set hlsearch                " Highlight searches by default.
set incsearch               " Incrementally search while typing a regex

" Removing tool bar
set guioptions-=T

" Setting cursor-line ON
set cursorline

" Setting number line ON
set number
set numberwidth=1

" Enable mouse support, unless in insert mode
set mouse=a
set ttymouse=xterm2

" Show title on console title bar
set title

" Undo
set noswapfile
set nobackup
set undodir=~/.vim/undodir
set undofile

" Replace the default grep program with ack
let g:ackprg="ag --column --color-match --line-numbers"
set grepprg=ag
set grepformat=%f:%l:%c:%m

" Auto change the directory to the current file I'm working on
"autocmd BufEnter * lcd %:p:h

" Insert completation
" don't select first item, follow typing in autocomplete
set completeopt=menuone,longest,preview
"" set pumheight=6                   "Keep a small completation window

filetype plugin indent on

""" Moving Around/Editing
set cursorline              " have a line indicate the cursor location$
set ruler                   " show the cursor position all the time
set nostartofline           " Avoid moving cursor to BOL when jumping around
set virtualedit=block       " Let cursor move past the last char in <C-v> mode
set scrolloff=3             " Keep 3 context lines above and below the cursor
set backspace=2             " Allow backspacing over autoindent, EOL, and BOL
set showmatch               " Briefly jump to a paren once it's balanced
set matchtime=2             " (for only .2 seconds).
set nowrap                  " don't wrap text
set linebreak               " don't wrap textin the middle of a word
set autoindent              " always set autoindenting on
set tabstop=4               " <tab> inserts 4 spaces
set shiftwidth=4            " but an indent level is 2 spaces wide.
"set softtabstop=4           " <BS> over an autoindent deletes both spaces.
set expandtab               " Use spaces, not tabs, for autoindent/tab key.
set shiftround              " rounds indent to a multiple of shiftwidthd
set matchpairs+=<:>         " show matching <> (html mainly) as well$
set foldmethod=indent       " allow us to fold on indents$
set foldlevel=99            " don't fold by default$

" close preview window automatically when we move around
autocmd CursorMovedI * if pumvisible() == 0|pclose|endif
autocmd InsertLeave * if pumvisible() == 0|pclose|endif


" Automatic closing brakets
imap { {}<left>
imap ( ()<left>
imap [ []<left>

" ===================================================================== "
" Setting this when need to log vim.
" mkdir ~/.log/vim
" ===================================================================== "
"set verbosefile=~/.log/vim/verbose.log
"set verbose=2
" ===================================================================== "

" try to detect filestypes
filetype on
" turn on synthax highlighting
syntax on


" ##### KEY MAPPING #####
let mapleader=","

" ,v  brings up my .vimrc
" ,V reload the .vimrc -- makign all change active (have to save first)
map <leader>v :sp ~/.vimrc<CR><C-W>_
map <silent> <leader>V :source ~/.vimrc<CR>:filetype detect<CR>:exe ":echo 'vimrc reloaded'"<CR>

" open/close the quick window$
nmap <leader>o :copen<CR>
nmap <leader>oo :cclose<CR>

" For when we forget to use sudo to open/edit a file
cmap w!! w !sudo tee % >/dev/null

" save file (ctrl-s)
map <C-s> :w<CR>

" copy selected text (ctrl-c)
vmap <C-c> +y

" paste clipboard contest (ctrl-v)
imap <C-p> <esc>P

" Open new file dialog (ctrl-n)
map <C-n> :browse confirm e<cr>

" Open save-as dialog (ctrl-shift-n)
map <C-S-s> :browse confirm saveas<cr>

" To open a new empty buffer
" This replaces :tabnew which I used to bind to this mapping
nmap <leader>T :enew<cr>

" Close all buffers 
nnoremap <silent> <leader>c :NERDTreeClose<bar>:BufOnly <CR>

" Buffer cycling
map <C-right> <ESC>:bn<cr>
map <C-left> <ESC>:bp<cr>

" Finding files with fzf
nnoremap <silent> <C-p> :Files<CR>
nnoremap <silent> <C-f> :Ag<CR>

command! -bang -nargs=* Ag
  \ call fzf#vim#ag(<q-args>,
  \                 <bang>0 ? fzf#vim#with_preview({'options': '--delimiter : --nth 4..'}, 'up:60%')
  \                         : fzf#vim#with_preview('right:50%:hidden', '?'),
  \                 <bang>0)

" Make <c-l> clear the highlight as well as redraw
nnoremap <C-L> :nohls<CR><C-L>
inoremap <C-L> <C-O>:nohls<CR>

" Key mapping for quickfix result navigation
"map <A-o> :copen<CR>
"map <A-q> :cclose<CR>
map <A-j> :cnext<CR>
map <A-k> :cprevious<CR>

" Key to remove duplicated lines.
map ,d <esc>:%s/\(^\n\{2,}\)/\r/g<CR>

" Ag searching
nmap <leader>a <ESC>:Ag!

" Mapping to move lines
nnoremap <C-j> :m+<CR>==
nnoremap <C-k> :m-2<CR>==
inoremap <C-j> <Esc>:m+<CR>==gi
inoremap <C-k> <Esc>:m-2<CR>==gi
vnoremap <C-j> :m'>+<CR>gv=gv
vnoremap <C-k> :m-2<CR>gv=gv

"key mapping for window navigation
"map <C-h> <C-w>h
"map <C-j> <C-w>j
"map <C-k> <C-w>k
"map <C-l> <C-w>l

"key mapping for tab navigation
" CTRL-Tab is next tab
noremap <C-Tab> :<C-U>tabnext<CR>
inoremap <C-Tab> <C-\><C-N>:tabnext<CR>
cnoremap <C-Tab> <C-C>:tabnext<CR>
" CTRL-SHIFT-Tab is previous tab
noremap <C-S-Tab> :<C-U>tabprevious<CR>
inoremap <C-S-Tab> <C-\><C-N>:tabprevious<CR>
cnoremap <C-S-Tab> <C-C>:tabprevious<CR>

" --------------------------------------------------------
" COC-VIM TAB SETTINGS START

" Use tab for trigger completion with characters ahead and navigate.
" NOTE: Use command ':verbose imap <tab>' to make sure tab is not mapped by
" other plugin before putting this into your config.
inoremap <silent><expr> <TAB>
      \ pumvisible() ? "\<C-n>" :
      \ <SID>check_back_space() ? "\<TAB>" :
      \ coc#refresh()
inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"

function! s:check_back_space() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

" Use <c-space> to trigger completion.
if has('nvim')
  inoremap <silent><expr> <c-space> coc#refresh()
else
  inoremap <silent><expr> <c-@> coc#refresh()
endif

" Use <cr> to confirm completion, `<C-g>u` means break undo chain at current
" position. Coc only does snippet and additional edit on confirm.
" <cr> could be remapped by other vim plugin, try `:verbose imap <CR>`.
if exists('*complete_info')
  inoremap <expr> <cr> complete_info()["selected"] != "-1" ? "\<C-y>" : "\<C-g>u\<CR>"
else
  inoremap <expr> <cr> pumvisible() ? "\<C-y>" : "\<C-g>u\<CR>"
endif

" Java
" Organize imports
nnoremap <silent> <leader>oi :call CocAction('runCommand', 'java.action.organizeImports')<CR>
"
" " Generate getters and setters 
nnoremap <silent> <leader>gs :call CocAction('runCommand', 'java.action.generateGettersAndSetters')<CR>
"
" " Generate equals and hashCode
nnoremap <silent> <leader>eh :call CocAction('runCommand', 'java.action.generateEqualsAndHashCode')<CR>
"
" " Rename symbol
nnoremap <silent> <leader>rn :call CocAction('rename')<CR>
"
" " Go to super implementation
nnoremap <silent> <leader>gi :call CocAction('runCommand', 'java.goto.superImplementation')<CR>
"
" " Start debugging
nnoremap <silent> <leader>db :call CocAction('runCommand', 'java.debug.start')<CR>

" COC-VIM TAB SETTINGS END
" --------------------------------------------------------

" Linting code
" Error and warning signs.
let g:ale_sign_error = '⤫'
let g:ale_sign_warning = '⚠'
" Enable integration with airline.
let g:airline#extensions#ale#enabled = 1

" ############ FUNCTIONS ###################

" reacalculate the traling whitespace warning when idle, and after saving
autocmd cursorhold,bufwritepost * unlet! b:statusline_trailing_space_warning

" Recalculate the tab warning flag when idle and after writing
autocmd cursorhold,bufwritepost * unlet! b:statusline_tab_warning

" Recalculate the long line warning whe idle and after saving
autocmd cursorhold,bufwritepost * unlet! b:statusline_long_line_warning

" Define :HighlightLongLines comman to highlight the offending parts of
" lines that are longer than the specified lgnth (defaulting to 100)
command! -nargs=? HighlihgtLongLines call s:HighlightLongLines('<args>')
function! s:HighlightLongLines(width)
    let targetWidth = a:width != '' ? a:width : 100
    if targetWidth > 0
        exec 'match Todo /\%>' . (targetWidth) . 'v/'
    else
        echomsg 'Usage: HighlightLongLines [length]"
    endif
endfunction

" Reaload .vimrc
" Soruce the .vimrc or _vimrc file
map ,v :e $HOME/.vimrc<CR>
nmap <F12> :<C-u>source .vimrc <BAR> echo "vimrc realoaded"<CR>

" ==================================================================
" Preserve
" =================================================================
function! Preserve(command)
    " Preparation: save last search, and cursor position.
    let _s=@/
    let l = line(".")
    let c = col(".")
    " Do the business:
    execute a:command
    " Clean up: restore previous search history, and cursor position
    let @/=_s
    call cursor(l,c)
endfunction

" ==================================================================
" Tidying Whitespaces
" ==================================================================
function! <SID>StripTrailingWhitespaces()
    :call Preserve("%s/\\s\\+$//e")
endfunction
nmap <leader>$ :call <SID>StripTrailingWhitespaces()<CR>

" ==================================================================
" Deleting blank lines
" ==================================================================
"Function: <SID>DeleteBlankLines
"Desc: delete blank lines on buffer
"
"Arguments:
"
function! <SID>DeleteBlankLines()
    :call Preserve(":g/^$/de")
endfunction
nmap <leader>0 :call <SID>DeleteBlankLines()<CR>

" ===================================================================
" GO
" ===================================================================
set autowrite

" Use deoplete.
let g:deoplete#enable_at_startup = 1

" Go syntax highlighting
let g:go_highilight_types = 1
let g:go_highlight_fields = 1
let g:go_highlight_functions = 1
let g:go_highlight_function_calls = 1
let g:go_highlight_extra_types = 1
let g:go_highlight_operators = 1
let g:go_highlight_build_contraints = 1

" Auto formatting and importing
let g:go_fmt_autosave = 1
let g:go_fmt_command = "goimports"

" Metalinter
let g:go_metalinter_autosave = 1
let g:go_metalinter_autosave_enabled = ['vet', 'golint', 'errcheck', 'deadcode']
let g:go_metalinter_deadline = "5s"

" Status line types/signatures
let g:go_auto_type_info = 1

" Quickfix
let g:go_list_type = "quickfix"

" Auto type info
let g:go_auto_type_info = 1

" Auto same ids
" let g:go_auto_sameids = 1

" Go rename
let g:go_rename_command = "gopls"

let g:ctrlp_buftag_types = {'go' : '--language-force=go --golang-types=ftv'}

" Run :GoBuild or :GoTestCompile based on the go file
function! s:build_go_files()
  let l:file = expand('%')
  if l:file =~# '^\f\+_test\.go$'
    call go#test#Test(0, 1)
  elseif l:file =~# '^\f\+\.go$'
    call go#cmd#Build(0)
  endif
endfunction

" Map keys for most used commands.
" Ex: `\b` for building, `\r` for running and `\b` for running test.
autocmd FileType go nmap <leader>b :<C-u>call <SID>build_go_files()<CR>
autocmd FileType go nmap <leader>r  <Plug>(go-run)
autocmd FileType go nmap <leader>t  <Plug>(go-test)
autocmd FileType go nmap <leader>i  <Plug>(go-info)
autocmd FileType go nmap <leader>d  <Plug>(go-doc)
autocmd BufNewFile,BufRead *.go setlocal noexpandtab tabstop=4 shiftwidth=4

au filetype go inoremap <buffer> . .<C-x><C-o>
au filetype go inoremap <C-@> <C-x><C-o>
" ===================================================================
" Python
" ===================================================================
autocmd FileType python set omnifunc=pythoncomplete#Complete
autocmd BufRead *py set efm=%C\ %.%#,%A\ \ File\ \"%f\"\\,\ line\ %l%.%#,%Z%[%^\ ]%\\@=%m

" Setting pylint
autocmd FileType python compiler pylint

" turn of hlsearch and update pyflakes on enter
"autocmd BufRead,BufNewFile *.py nnoremap <buffer><CR> :nohlsearch\|:call PressedEnter()<cr>
nnoremap <buffer><CR> :nohlsearch \| :call PressedEnter()<cr>

" Clear the search buffer when hitting return and update pyflake checks
function! PressedEnter()
    :nohlsearch
endfunction

if has("autocmd")
"   autocmd FileType python autocmd BufWritePre <buffer> :call Autopep8()
"   autocmd FileType python autocmd BufWritePre <buffer> execute ':Black'
    autocmd FileType python autocmd BufWritePre <buffer> execute ':Ruff'
    autocmd FileType c,python,java,javascript,html,css autocmd BufWritePre <buffer> :call <SID>StripTrailingWhitespaces()
    autocmd FileType html,css autocmd BufWritePre <buffer> :call <SID>DeleteBlankLines()
    autocmd BufReadPost fugitive://* set bufhidden=delete
endif

" ############### PLUGINS SETTINGS ##################

" Ariline
" "----------------------------------------------
" Plugin: bling/vim-airline
"----------------------------------------------
" Show status bar by default.
set laststatus=2

" Theme
let g:airline_theme='badwolf'

" Show full path
let g:airline_section_c = '%F'

" Enable top tabline.
let g:airline#extensions#tabline#enabled = 1

let g:airline#extensions#tabline#left_sep = ' '
let g:airline#extensions#tabline#left_alt_sep = '|'

" Disable showing tabs in the tabline. This will ensure that the buffers are
" what is shown in the tabline at all times.
let g:airline#extensions#tabline#show_tabs = 0

" Enable powerline fonts.
let g:airline_powerline_fonts = 0

" Explicitly define some symbols that did not work well for me in Linux.
if !exists('g:airline_symbols')
    let g:airline_symbols = {}
endif
let g:airline_symbols.branch = ''
let g:airline_symbols.maxlinenr = ''

let g:airline#extensions#tabline#fnamemod = ':t'

" Autopep8
" Disable show diff window
let g:autopep8_disable_show_diff=1
let g:autopep8_max_line_length=79

" Fuzzy Finder Settings
let g:fuzzy_matching_limit = 20
let g:fuzzy_ignore="*.ico;*.png;*PNG;*.jpg;*.JPG;*.GIF;*.gif;tmp/**;log/**"

" disable caching for Fuzzy Finder
let g:fuf_tag_cache_dir = ''
let g:fuf_taggedfile_cache_dir = ''

" NERDTree
map <C-z> :NERDTreeToggle<CR> 
let g:NERDTreeDirArrowExpandable = '▸'
let g:NERDTreeDirArrowCollapsible = '▾'
let NERDTreeShowHidden=1 
let NERDTreeIgnore=['\.git$', '\.idea$', '\.vscode$', '\.history$']

" Coc
set hidden
set cmdheight=2
set updatetime=300
set shortmess+=c
if has("patche-8.1.1564")
    set signcolumn=number
else
    set signcolumn=yes
endif

nmap <silent> gr <Plug>(coc-references)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> rn <Plug>(coc-rename)

nnoremap <silent> K :call <SID>show_documentation()<CR>
function! s:show_documentation()
  if (index(['vim','help'], &filetype) >= 0)
    execute 'h '.expand('<cword>')
  else
    call CocAction('doHover')
  endif
endfunction

" Highlight the symbol and its references when holding the cursor.
autocmd CursorHold * silent call CocActionAsync('highlight')

" Formatting selected code.
xmap <leader>f  <Plug>(coc-format-selected)
nmap <leader>f  <Plug>(coc-format-selected)

""" Coc - Customize colors
func! s:my_colors_setup() abort
    " this is an example
    hi Pmenu guibg=#d7e5dc gui=NONE
    hi PmenuSel guibg=#b7c7b7 gui=NONE
    hi PmenuSbar guibg=#bcbcbc
    hi PmenuThumb guibg=#585858
endfunc

augroup colorscheme_coc_setup | au!
    au ColorScheme * call s:my_colors_setup()
augroup END

""" Ultisnip
let g:UltiSnipsExpandTrigger = "<tab>"
"let g:UltiSnipsExpandTrigger="<Nop>"

"" Setting background
set background=dark
let g:rehash256 = 1
let g:one_allow_italics = 1
"let g:molokai_original = 1
colorscheme jellybeans

"highlight Pmenu ctermfg=7 ctermbg=242 guifg=LightGrey guibg=DarkGrey

function! CloseOtherBuffer()
    let l:bufnr = bufnr()
    execute "only"
    for buffer in getbufinfo()
        if !buffer.listed
            continue
        endif
        if buffer.bufnr == l:bufnr
            continue
        else
            if buffer.changed
                echo buffer.name . " has changed, save first"
                continue
            endif
            let l:cmd = "bdelete " . buffer.bufnr
            execute l:cmd
        endif
    endfor
endfunction
nnoremap <leader>x :call CloseOtherBuffer()<CR>
