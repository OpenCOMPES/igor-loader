import numpy as np
import xarray as xr
from igor2 import binarywave
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
from pathlib import Path

class static_analysis:
    
    def load(self,path_ibw,name,run):
        path = Path(path_ibw)
        files = sorted(path.glob(f"{name}{run:03d}*.ibw"))
        if len(files)==0:
            raise ValueError("this file does not exist")
        else:
            if len(files)==1:
                self.data = self.ibw_to_xarray(files[0])
                self.edit()
            else:
                self.axis={}
                for i, file in enumerate(files):
                    self.ibw_to_xarray(file)
                    self.edit()
                    self.axis[str(i+1)] = self.data
            
                    # data = binarywave.load(files[i])
                    # metadata = self.build_metadata_from_ibw(data)
                    # phi=metadata['sample']['orientation']['Phi'] #Identify the coordinate of the axis scan and add to the coordinates.
            
                    # self.phis.append(phi) 
                    # self.tilt[str(i-fst)]['Phi']=metadata['sample']['position']['Phi']
                
        # return self.data
            
            
        
        
    def ibw_to_xarray(self,path):
        
        bw = binarywave.load(path)
    
        wave = bw["wave"]
        header = wave["wave_header"]
        data = wave["wData"]
    
        ndim = data.ndim
    
        sfA = header["sfA"]
        sfB = header["sfB"]
        nDim = header["nDim"]
    
        coords = {}
        dims = []
        metadata = self.build_metadata_from_ibw(bw)
    
        for i in range(ndim):
            n = nDim[i]
            start = sfB[i]
            delta = sfA[i]
    
            coord = start + delta * np.arange(n)
    
            dim_name = f"dim_{i}"
            dims.append(dim_name)
            coords[dim_name] = coord
    
        coords['X'] = metadata['sample']['position']['X']
        coords['Y'] = metadata['sample']['position']['Y']
        coords['Z'] = metadata['sample']['position']['Z']
        coords['Phi'] = metadata['sample']['position']['Phi']
        coords['Theta'] = metadata['sample']['position']['Theta']
        coords['Omega'] = metadata['sample']['position']['Omega']
        
        
        self.data = xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            name=header["bname"].decode("ascii"),
            attrs={
                "note": wave["note"].decode("latin1", errors="ignore")
            }
        )
    
        return self.data


    def load_old(self,path_ibw):
                 
        bw = binarywave.load(path_ibw)
    
        wave = bw["wave"]
        header = wave["wave_header"]
        data = wave["wData"]
    
        ndim = data.ndim
    
        sfA = header["sfA"]
        sfB = header["sfB"]
        nDim = header["nDim"]
    
        coords = {}
        dims = []
    
        for i in range(ndim):
            n = nDim[i]
            start = sfB[i]
            delta = sfA[i]
    
            coord = start + delta * np.arange(n)
    
            dim_name = f"dim_{i}"
            dims.append(dim_name)
            coords[dim_name] = coord
    
        self.data = xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            name=header["bname"].decode("ascii"),
            attrs={
                "note": wave["note"].decode("latin1", errors="ignore")
            }
        )
    
        return self.data
    
    def parse_ibw_note(self,ibw_dict):
        """
        Parse the IGOR wave note into a Python dictionary.
        """
        note_raw = ibw_dict['wave']['note']
    
        # Decode if necessary
        if isinstance(note_raw, bytes):
            note_raw = note_raw.decode('utf-8', errors='ignore')
    
        note_dict = {}
    
        # Common separators: newline or semicolon
        for line in note_raw.replace(';', '\n').splitlines():
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
    
            # Try to cast to float or int when possible
            try:
                if '.' in value or 'e' in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
    
            note_dict[key] = value
    
        return note_dict
    
    
    
    def build_metadata_from_ibw(self,ibw_dict):
        note = self.parse_ibw_note(ibw_dict)
    
        metadata = {}
    
        # ======================
        # General
        # ======================
        wave_name = ibw_dict['wave']['wave_header']['bname'].decode('utf-8', errors='ignore')
        metadata['entry_title'] = wave_name
    
        # ======================
        # Instrument
        # ======================
        metadata['instrument'] = {}
    
        # ----------------------
        # Beam (probe)
        # ----------------------
        metadata['instrument']['beam'] = {}
        metadata['instrument']['beam']['probe'] = {}
    
        # Generic energy mappings
        metadata['instrument']['beam']['probe']['incident_energy'] = note.get('Eph')
        metadata['instrument']['beam']['probe']['electron_kinetic_energy'] = note.get('Ek')
        metadata['instrument']['beam']['probe']['pass_energy'] = note.get('Ep')
    
        # Optional / may not exist
        metadata['instrument']['beam']['probe']['pulse_duration'] = note.get('PulseDuration')
        metadata['instrument']['beam']['probe']['frequency'] = note.get('Frequency')
        metadata['instrument']['beam']['probe']['incident_polarization'] = None
        metadata['instrument']['beam']['probe']['extent'] = None
    
        # ----------------------
        # Analyzer
        # ----------------------
        metadata['instrument']['electronanalyzer'] = {
            'lens_mode': note.get('LensMode'),
            'energy_resolution': None,
            'angular_resolution': None,
            'spatial_resolution': None,
        }
    
        # ----------------------
        # Detector / acquisition
        # ----------------------
        metadata['instrument']['detector'] = {
            'Udet': note.get('Udet'),
            'Uscr': note.get('Uscr'),
            'Icoil': note.get('Icoil'),
        }
    
        metadata['instrument']['acquisition'] = {
            'dwell_time': note.get('DwellTime'),
            'exposures': note.get('Exposures'),
            'num_images': note.get('NumOfImages'),
            'pixel_sum': note.get('PixelSum'),
        }
    
        # ----------------------
        # Vacuum
        # ----------------------
        metadata['instrument']['vacuum_pressure'] = note.get('MCPress')
    
        # ======================
        # Sample
        # ======================
        metadata['sample'] = {}
    
        # Temperatures (generic handling)
        temperatures = {}
        for key in note:
            if key.upper().startswith('T_'):
                temperatures[key] = note[key]
        if temperatures:
            metadata['sample']['temperature'] = temperatures
    
        # Position
        metadata['sample']['position'] = {
            'X': note.get('X'),
            'Y': note.get('Y'),
            'Z': note.get('Z'),
            'Theta': note.get('Theta'),
            'Phi': note.get('Phi'),
            'Omega': note.get('Omega')
        }
    
        # ======================
        # ROI
        # ======================
        roi_keys = ('ROIX1', 'ROIX2', 'ROIY1', 'ROIY2')
        if any(k in note for k in roi_keys):
            metadata['instrument']['roi'] = {
                'x1': note.get('ROIX1'),
                'x2': note.get('ROIX2'),
                'y1': note.get('ROIY1'),
                'y2': note.get('ROIY2'),
            }
    
        # ======================
        # Raw note (full provenance)
        # ======================
        metadata['raw_note'] = note
    
        return metadata
    
    def edit(self,new_dims=None):
        if new_dims is None:
            new_dims = ['Energy', 'Angular']
        if len(new_dims) != len(self.data.dims):
            raise ValueError("Number of new dimensions must match existing ones")
        self.data=self.data.rename(dict(zip(self.data.dims, new_dims)))
        # return self.data.rename(dict(zip(self.data.dims, new_dims)))
    
    
    def load_tilt_old(self,datapath,fst,last):
        self.tilt = {}
        self.phis=[]
        
        for i in range(fst, last):
            self.load_old(datapath + f"WSe2_{i:03d}.ibw")
            self.edit()
            self.tilt[str(i-fst)] = self.data
            
            data = binarywave.load(datapath + f"WSe2_{i:03d}.ibw")
            metadata = self.build_metadata_from_ibw(data)
            phi=metadata['sample']['position']['Phi']
            
            self.phis.append(phi) 
            self.tilt[str(i-fst)]['Phi']=metadata['sample']['position']['Phi']

    
    def combine_tilts(self):
        """
        tilt : dict-like (ex: tilt["0"], tilt["1"], ...)
        phis : array-like com os valores de Phi
        scan_range : iterable (ex: range(0, 21))
        """
        scans = [self.tilt[str(i)] for i in range(0,len(self.tilt))]
    
        stacked = xr.concat(
            scans,
            dim=xr.DataArray(self.phis, dims="scan", name="Phi")
        )
    
        stacked = stacked.set_index(scan="Phi")
        stacked = stacked.rename(scan="Phi")
    
        return stacked
    
    def tilt_map(self,stacked):
    
        phi_vals = stacked.coords["Phi"].values
        ang_vals = stacked.coords["Angular"].values
        energy_vals = stacked.coords["Energy"].values
    
        def update(Energy=1, dE=0.5, Phi=0, dPhi=0, Angular=0, dang=0):
    
            # ---------- Dados ----------
            data_main = stacked.sel(
                Energy=slice(Energy - dE, Energy + dE)
            ).mean(dim="Energy")
    
            data_phi = stacked.sel(
                Phi=slice(Phi - dPhi, Phi + dPhi)
            ).mean(dim="Phi")
    
            data_ang = stacked.sel(
                Angular=slice(Angular - dang, Angular + dang)
            ).mean(dim="Angular")
    
            # ---------- Figura ----------
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
            # --- Mapa principal (Phi x Angular) ---
            im0 = axes[0].imshow(
                data_main,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[
                    ang_vals.min(), ang_vals.max(),
                    phi_vals.min(), phi_vals.max()
                ]
            )
    
            axes[0].axhspan(Phi - dPhi, Phi + dPhi, color="r", alpha=0.2)
            axes[0].axvspan(Angular - dang, Angular + dang, color="b", alpha=0.2)
    
            axes[0].set_xlabel("Angular")
            axes[0].set_ylabel("Phi")
            axes[0].set_title(f"Energy ∈ [{Energy-dE:.2f}, {Energy+dE:.2f}]")
            fig.colorbar(im0, ax=axes[0])
    
            # --- Corte Phi fixo (Energy x Angular) ---
            im1 = axes[1].imshow(
                data_phi,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[
                    ang_vals.min(), ang_vals.max(),
                    energy_vals.min(), energy_vals.max()
                ]
            )
            axes[1].set_xlabel("Angular")
            axes[1].set_ylabel("Energy")
            axes[1].set_title(f"Phi = {Phi:.2f}")
            fig.colorbar(im1, ax=axes[1])
    
            # --- Corte Angular fixo (Energy x Phi) ---
            im2 = axes[2].imshow(
                data_ang,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[
                    phi_vals.min(), phi_vals.max(),
                    energy_vals.min(), energy_vals.max()
                ]
            )
            axes[2].set_xlabel("Phi")
            axes[2].set_ylabel("Energy")
            axes[2].set_title(f"Angular = {Angular:.2f}")
            fig.colorbar(im2, ax=axes[2])
    
            plt.tight_layout()
            plt.show()
    
        # ---------- Sliders ----------
        Energy_slider = widgets.FloatSlider(
            value=energy_vals.mean(),
            min=energy_vals.min(),
            max=energy_vals.max(),
            step=np.diff(energy_vals).mean(),
            description='Energy'
        )
    
        dE_slider = widgets.FloatSlider(value=0.5, min=0.1, max=2, step=0.1, description='dE')
        dPhi_slider = widgets.FloatSlider(value=1, min=0.1, max=5, step=1, description='dPhi')
        dang_slider = widgets.FloatSlider(value=1, min=1, max=10, step=1, description='dang')
    
        Phi_slider = widgets.FloatSlider(
            value=phi_vals[len(phi_vals)//2],
            min=phi_vals.min(),
            max=phi_vals.max(),
            step=np.diff(phi_vals).mean(),
            description='Phi'
        )
    
        Angular_slider = widgets.FloatSlider(
            value=ang_vals[len(ang_vals)//2],
            min=ang_vals.min(),
            max=ang_vals.max(),
            step=np.diff(ang_vals).mean(),
            description='Angular'
        )
    
        interact(
            update,
            Energy=Energy_slider,
            dE=dE_slider,
            dPhi=dPhi_slider,
            dang=dang_slider,
            Phi=Phi_slider,
            Angular=Angular_slider
        )

    def edc(self):
        data = self.data

        # Coordenadas
        x = data.coords[data.dims[1]].values  # Angular
        y = data.coords[data.dims[0]].values  # Energy

        # -------- Sliders --------
        c_slider = widgets.FloatSlider(
            value=float(x[len(x)//2]),
            min=float(x.min()),
            max=float(x.max()),
            step=float(np.diff(x).mean()),
            description=data.dims[1]
        )

        cut_slider = widgets.FloatSlider(
            value=float(y.min()),
            min=float(y.min()),
            max=float(y.max()),
            step=float(np.diff(y).mean()),
            description="Cut"
        )

        end_slider = widgets.FloatSlider(
            value=float(y.max()),
            min=float(y.min()),
            max=float(y.max()),
            step=float(np.diff(y).mean()),
            description="End"
        )

        # -------- Função interna --------
        def update(c, cut, end):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # ----- Mapa 2D -----
            img_cut = data.sel({data.dims[0]: slice(cut, end)})

            im = axes[0].pcolormesh(
                img_cut.coords[data.dims[1]],
                img_cut.coords[data.dims[0]],
                img_cut.values,
                shading="auto"
            )

            axes[0].axvline(c, color="k", lw=3)
            axes[0].set_title("Intensity Map")
            axes[0].set_xlabel(data.dims[1])
            axes[0].set_ylabel(data.dims[0])
            fig.colorbar(im, ax=axes[0])

            # ----- EDC -----
            edc = (
                data
                .sel({data.dims[1]: c}, method="nearest")
                .sel({data.dims[0]: slice(cut, end)})
            )

            axes[1].plot(
                edc.coords[data.dims[0]],
                edc.values,
                label=f"{data.dims[1]} ≈ {c:.2f}"
            )

            axes[1].set_title("EDC")
            axes[1].set_xlabel("Energy (eV)")
            axes[1].set_ylabel("Intensity")
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        # -------- UI --------
        ui = widgets.VBox([c_slider, cut_slider, end_slider])
        out = widgets.interactive_output(
            update,
            {"c": c_slider, "cut": cut_slider, "end": end_slider}
        )

        display(ui, out)